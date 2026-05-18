"""Per-split-point energy and latency profiling (data-owner side).

Helpers used by the client worker to estimate the power consumption and
wall-clock cost of training the local sub-model at each candidate split
layer, given the raw power logs collected by KASA or ``jtop``.
"""

import os,math,numpy as np,datetime
import pandas as pd
csvPath="CSVResults/"


def CombineLogCSV(name="192.168.4.78", sl=3, maxepoch=None, outfolder=None):
	try:
		basefolder = csvPath
		if outfolder is not None:
			os.makedirs(outfolder, exist_ok=True)
		with open(basefolder + "Power_" + name + ".csv") as f:
			columnlen=len(f.readline().split(","))
		client_log = pd.read_csv(basefolder + "Log_" + name + ".csv", names=["Time", "Message"])
		client_log["Time"] = client_log["Time"].map(pd.to_datetime)
		power_log = pd.read_csv(basefolder + "Power_" + name + ".csv",names=["Time", "Watts", "Stime", "Ram"]+ ["CPU" + str(i) for i in range(1, columnlen-3)])
		power_log["Time"] = power_log["Time"].map(pd.to_datetime)
		df = pd.merge(power_log, client_log, on='Time', how='outer')
		df = df.sort_values("Time").reset_index()
		del df["index"]
		# df.at[0, 'Message']="Nothing"
		prevrow = df.iloc[0].copy()
		if isinstance(prevrow["Message"], float) and math.isnan(prevrow["Message"]):
			prevrow['Message'] = "Nothing"
		rowlist = []
		for i, row in df.iterrows():
			# if math.isnan(row["Watts"]):
			# 	row["Watts"]=prevrow["Watts"]
			if isinstance(row["Message"], float) and math.isnan(row["Message"]):
				message = prevrow["Message"]
				if "Start" in message: message = message.replace("Start", "Continue")
				if "Finish" in message: message = "Nothing"
				if "Add Schedular step" == message: message = "Nothing"
				row["Message"] = message
			prevrow = row
			rowlist.append(row)
		df = pd.DataFrame(rowlist)
		df = df.interpolate().bfill()
		df["Client"] = name
		df = df[["Time", "Client", "Watts", "Message", ]]
		del df["Client"]
		fullshape = df.shape[0]
		if maxepoch is not None:
			expname = "_".join(name.split("_")[:6])
			endmessage = f"Training for Epoch {maxepoch} experiment {expname} Finish"
			df = df[:df[df["Message"] == endmessage].index[0] + 1]
		if outfolder is not None:
			df.to_csv(f"{outfolder}Power_{sl}_{name}.csv", index=None)
			print(f"Saved at {outfolder}Power_{sl}_{name}.csv")
		return df
	except Exception as e:
		print("Exception ", e, "In file", name)
		raise e

def getAggerigatedTimeFor(df, Message,aggtype="min"):
	if aggtype=="min":
		return df["Watts"].min()
	else:
		df = df[df["Message"] == Message]
		return df["Watts"].mean()

def combinecontinuesevnts(df,startidx,endidx):
	df['Time'] = pd.to_datetime(df['Time'])
	dfmessages=df["Message"]
	sidx,eidx=startidx[0],endidx[0]
	newstartidx,newendidx=[],[]
	prevendtime=df.loc[startidx[0],"Time"]
	if len(startidx)==len(endidx):
		for s,e in zip(startidx,endidx):
			stime,etime = df.loc[s, "Time"],df.loc[e, "Time"]
			if (stime-prevendtime).seconds==0:
				eidx=e
			else:
				newstartidx.append(sidx)
				newendidx.append(eidx)
				sidx,eidx=s,e
			prevendtime=etime
		newstartidx.append(sidx)
		newendidx.append(eidx)
	else:
		si,ei=0,0
		while si<len(startidx) and ei<len(endidx):
			s,e=startidx[si],endidx[si]
			while s>e and ei<len(endidx)-1:
				ei+=1
				e=endidx[ei]
			while si<len(startidx)-1 and startidx[si+1]<e:
				si+=1
				s=startidx[si]
			stime,etime = df.loc[s, "Time"],df.loc[e, "Time"]
			if (stime-prevendtime).seconds==0:
				eidx=e
			else:
				newstartidx.append(sidx)
				newendidx.append(eidx)
				sidx,eidx=s,e
			prevendtime=etime
			si,ei=si+1,ei+1
		if sidx<eidx:
			newstartidx.append(sidx)
			newendidx.append(eidx)

	return np.array(newstartidx),np.array(newendidx)


def getCommunicationStartEnd(df):
	comnstartindex=df[df["Message"].str.contains("Send MidOutput Start")|df["Message"].str.contains("Send Gradient Start")].index
	comnendindex=df[df["Message"].str.contains("Send MidOutput Finish")|df["Message"].str.contains("Send Gradient Finish")].index
	return combinecontinuesevnts(df,comnstartindex,comnendindex)

def getComputationStartend(df):
	# compstartindex=df[df["Message"].str.contains("Loading Data Start")|df["Message"].str.contains("Forward Propagation Start")|df["Message"].str.contains("Update Gradient Start")].index
	# compendindex=df[df["Message"].str.contains("Loading Data Finish")|df["Message"].str.contains("Forward Propagation Finish")|df["Message"].str.contains("Update Gradient Finish")].index
	mLabels=["Loading Data","Update Gradient","Forward Propagation","Client Decomposition","Client Reconstruction"]
	# mLabels=["Client Decomposition","Client Reconstruction"]
	compstartFlag,compendFlag=[],[]
	for m in mLabels:
		compstartFlag.append(df["Message"].str.contains(m+" Start"))
		compendFlag.append(df["Message"].str.contains(m+" Finish"))
		if len(compstartFlag)>1:
			compstartFlag=[np.bitwise_or(*compstartFlag)]
			compendFlag = [np.bitwise_or(*compendFlag)]
	compstartindex=df[compstartFlag[0]].index
	compendindex=df[compendFlag[0]].index
	return combinecontinuesevnts(df,compstartindex,compendindex)

def getIdleStartEnd(df):
	indexflag=df["Message"] == "Nothing"
	starttimes,endtime=[],[]
	prev=False
	for i,cur in enumerate(indexflag):
		if cur and not prev:
			starttimes.append(i)
		elif prev and not cur:
			endtime.append(i-1)
		prev=cur
	if prev:
		endtime.append(i)
	return combinecontinuesevnts(df,np.array(starttimes),np.array(endtime))


def getEnegryConsumption(df,contype="communication",basepower=0):
	df = df.ffill()
	if df.shape[0]==0: return None,None
	df['Time'] = pd.to_datetime(df['Time'])
	df.dropna(inplace=True)
	if contype == "communication":
		startindex,endindex=getCommunicationStartEnd(df)
	elif contype=="computation":
		startindex,endindex=getComputationStartend(df)
	elif contype=="idle":
		startindex,endindex=getIdleStartEnd(df)
	assert startindex.shape[0]==endindex.shape[0],"Start and end Index should have same shape for "+contype+" in Dataframe"
	btime = (df["Time"]-df["Time"].shift(1)).bfill()
	df["seconds"]=btime.map(lambda x:(x.seconds*1e+6+x.microseconds)/1e+6)
	if contype=="idle":
		df["PowerComsumption"]=df["seconds"]*df["Watts"]
	else:
		df["PowerComsumption"]=df["seconds"]*(df["Watts"]-basepower)
	# df.set_index('Time', inplace=True)
	# total=np.sum([df[s:e+1]["Watts"].resample("1S").mean().sum() for s,e in zip(startindex,endindex)])
	# mean=np.mean([df[s:e+1]["Watts"].resample("1S").mean().mean() for s,e in zip(startindex,endindex)])
	total=np.sum([df[s:e+1]["PowerComsumption"].sum() for s,e in zip(startindex,endindex)])
	mean=np.mean([df[s:e+1]["PowerComsumption"].mean() for s,e in zip(startindex,endindex)])
	max=np.max([df[s:e+1]["PowerComsumption"].max() for s,e in zip(startindex,endindex)])
	totaltime=np.sum([df.loc[e+1,"Time"]-df.loc[s,"Time"] for s,e in zip(startindex,endindex)])
	if contype=="idle" and basepower!=0:
		totaltime=df.loc[df.shape[0]-1,"Time"]-df.loc[0,"Time"]
		totalusedtimes=sum([df.loc[e,"Time"]-df.loc[s,"Time"] for s, e in zip(startindex, endindex)],datetime.timedelta())
		total=total /  totalusedtimes.seconds*totaltime.seconds
	return total,mean,max,totaltime

def getTotalDataTransfered(df):
	messages=df["Message"]
	messages=messages[messages.str.contains("MB Data")].map(lambda x:float(x.split()[-3]))
	return messages.sum()

def getPowerConsumption(experiment,splitpoint,subtractIdle=False):
	# maxepoch=min(max(10, k * 3), 100)
	# print(f,s)
	file=[file.split(".")[0][4:] for file in os.listdir(csvPath) if file.startswith("Log_" + experiment)][0]
	df=CombineLogCSV(file, splitpoint)
	if subtractIdle:
		idealpower = getAggerigatedTimeFor(df, Message="Nothing")
	else:
		idealpower = 0
	totalData=getTotalDataTransfered(df)
	cummnication,_,maximum1,cummtime = getEnegryConsumption(df, contype="communication", basepower=idealpower)
	compution,_,maximum2,comptime = getEnegryConsumption(df, contype="computation", basepower=idealpower)
	return cummnication,compution,cummtime,comptime,totalData,np.max((maximum1,maximum2))

def SplitPointProfileing(explist,splitpoints,subtractIdle=False):
	EnergyInfo=[]
	for exp, s in zip(explist, splitpoints):
		powerconsumption=getPowerConsumption(exp,s,subtractIdle=subtractIdle)
		EnergyInfo.append((exp, s, *powerconsumption))
	df=pd.DataFrame(EnergyInfo,columns=["Experiment","Split Layer","Communication","Computation","TotalCommunicationTime", "TotalComputationTime","TotalData(MB)","MaxEnergy"])
	df["Total"]=df["Communication"]+df["Computation"]
	df.to_csv("SplitProfiling.csv",index=None)
	print(df)
	return df

def getOptimimalSplitPoint(client,noicelevel,maxpoints,alpha=0.5):
	fsimarray = pd.read_csv("Fsimmean.csv")
	fsimarray = fsimarray.set_index("SplitLayer")
	df=pd.read_csv(f"EnergyCsv/{client}.csv")
	df=df[["Split Layer","Total"]]
	totalmax,totalmin=df["Total"].max(),df["Total"].min()
	df["Total"]=(df["Total"]-totalmin)/(totalmax-totalmin) if totalmax != totalmin else 0
	fsimmax,fsimmin=fsimarray.max(),fsimarray.min()
	fsimarray=(fsimarray-fsimmin)/(fsimmax-fsimmin)
	df=df[:maxpoints]
	energyconsumptins=[(i,df1["Split Layer"],df1["Total"]) for i,df1 in df.iterrows()]
	minimum=min(energyconsumptins,key=lambda x:x[2])
	subsetenrgy=energyconsumptins[minimum[0]:]
	ranklist=[]
	for i,sp,engcons in subsetenrgy:
		ranklist.append((int(sp),alpha*fsimarray.loc[int(sp),str(noicelevel)]+(1-alpha)*engcons))
	ranklist=sorted(ranklist,key=lambda x:x[1])
	return [s for s,r in ranklist]


if __name__ == '__main__':
	os.chdir("../ServerResults")
	csvPath="CSV/"
	experiment="24_04_10_05_34_14"
	splitpoint=2,5,8,10
	print(getPowerConsumption(experiment,splitpoint[0]))
	# print(SplitPointProfileing([experiment], splitpoint))