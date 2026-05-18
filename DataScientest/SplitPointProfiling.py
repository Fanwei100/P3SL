"""Optimal split-point selection from profiled energy logs.

Coordinator-side analytics over the per-split-point CSVs collected from each
client worker. The public helpers are:

* :func:`calculatePowerConsumptions` -- aggregate raw KASA / jtop samples
  into per-phase energy estimates;
* :func:`SplitPointProfileingLocal` -- summarise local profiling results;
* :func:`getOptimimalSplitPoint` -- choose the split layer that minimises
  the configured energy/feature-similarity trade-off;
* :func:`CalculateSplitPoints` -- batch-driver used by the experiments.
"""

import os,math,numpy as np,datetime
from Util import SplitProfilingfFolder,LoadKasaLogs
import pandas
import pandas as pd
csvPath="CSV/"

def CombineEnegryLogAndCodeLog(power_log,client_log,name, sl=3, maxepoch=None, outfolder=None,FORMAT="JTOP",DROPExtra=True):
	# try:
		if outfolder is not None:
			os.makedirs(outfolder, exist_ok=True)
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
	# except Exception as e:
	# 	print("Exception ", e, "In file", name)
	# 	raise e


def getAggerigatedTimeFor(df, Message,aggtype="min"):
	if aggtype=="min":
		return df["Watts"].min()
	else:
		df = df[df["Message"] == Message]
		return df["Watts"].mean()

def CombineLogCSV(name="192.168.4.78", sl=3, maxepoch=None, outfolder=None,DROPExtra=False):
	basefolder = "CSV/"
	enrgyFile=basefolder + "Power_" + name + ".csv"
	LogFile=basefolder + "Log_" + name + ".csv"
	with open(enrgyFile) as f:
		columnlen = len(f.readline().split(","))
	client_log = pd.read_csv(LogFile, names=["Time", "Message"])
	client_log["Time"] = client_log["Time"].map(pd.to_datetime)
	power_log = pd.read_csv(enrgyFile, names=["Time", "Watts", "Stime", "Ram"] + ["CPU" + str(i) for i in range(1, columnlen - 3)])
	power_log["Time"] = power_log["Time"].map(pd.to_datetime)
	if DROPExtra:
		expStart, expEnd = client_log.loc[0, "Time"] - pandas.Timedelta(10, "s"), client_log.loc[
			client_log.index[-1], "Time"] + pandas.Timedelta(10, "s")
		power_log = power_log[(expStart < power_log["Time"]) & (expEnd > power_log["Time"])]
	power_log["Time"] = power_log["Time"].map(pd.to_datetime)

	return CombineEnegryLogAndCodeLog(power_log,client_log, name, sl=sl, maxepoch=maxepoch, outfolder=outfolder)

def getTotalDataTransfered(df):
	messages=df["Message"]
	messages=messages[messages.str.contains("MB Data")].map(lambda x:float(x.split()[-3]))
	return messages.sum()

def calculatePowerConsumptions(clientName,expariment,clientLog,sl=3,subtractIdle=False,DROPExtra=True,SaveLogsLocally=True):
	print(f"calculatePowerConsumptions {clientName} {expariment}")
	try:
		clientLog,power_log=LoadKasaLogs(clientName,clientLog,expariment,SaveLogsLocally=SaveLogsLocally,DROPExtra=DROPExtra)
		df=CombineEnegryLogAndCodeLog(power_log,clientLog,name=expariment,sl=sl,FORMAT="KASA")
		if subtractIdle:
			idealpower = getAggerigatedTimeFor(df, Message="Nothing")
		else:
			idealpower = 0
		totalData = getTotalDataTransfered(df)
		df.to_csv("Temp.csv")
		cummnication, _, maximum1, cummtime = getEnegryConsumption(df, contype="communication", basepower=idealpower)
		compution, _, maximum2, comptime = getEnegryConsumption(df, contype="computation", basepower=idealpower)
		# idle, _, maximum3, idletime = getEnegryConsumption(df, contype="idle", basepower=0)
		return cummnication, compution, cummtime, comptime, totalData, np.max((maximum1, maximum2))
	except Exception as e:
		print("Exception in calculatePowerConsumptions Different timezone can be reason",e)
		raise e

def SplitPointProfileingLocal(explist,splitpoints):
	print("SplitPointProfileingLocal",explist,splitpoints)
	DfList=[]
	for exp, s in zip(explist, splitpoints):
		df=pd.read_csv(f"{SplitProfilingfFolder}{exp}.csv")
		DfList.append(df)
	df=pd.concat(DfList)
	df["Total"]=df["Communication"]+df["Computation"]
	return df


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


def getAverageTimeFor(df, Message):
	df=df[df["Message"]==Message]
	return df["Watts"].min()

def combinecontinuesevnts(df,startidx,endidx):
	df['Time'] = pd.to_datetime(df['Time'])
	sidx,eidx=startidx[0],endidx[0]
	newstartidx,newendidx=[],[]
	prevendtime=df.loc[startidx[0],"Time"]
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



def SplitPointProfileingWorker(explist,worker,subtractIdle=True,index=0):
	splitpoints,files=[],[]
	for expname in explist:
		df = pd.read_csv("Results/Result_" + expname+".csv")
		# print(df["Worker"].unique())
		# print(worker)
		splitpoints.append(df[df["Worker"]==worker]["Split layer"].values[0])
		# k = df.loc[0, "AggregationEpoch"]
		files.append([file.split(".")[0][4:] for file in os.listdir(csvPath) if file.startswith("Log_" + expname)][0])
	EnergyInfo=[]
	for f, s in zip(files, splitpoints):
		# maxepoch=min(max(10, k * 3), 100)
		# print(f,s)
		df=CombineLogCSV(f, s)
		print(s,df.shape)
		if subtractIdle:
			averageIdealpower = getAverageTimeFor(df, Message="Nothing")
		else:
			averageIdealpower = 0
		cummnication = getEnegryConsumption(df, contype="communication", basepower=averageIdealpower)[index]
		compution = getEnegryConsumption(df, contype="computation", basepower=averageIdealpower)[index]
		print(f,s,averageIdealpower,compution,cummnication)
		EnergyInfo.append((s,cummnication,compution))

	df=pd.DataFrame(EnergyInfo,columns=["Split Layer","Communication","Computation"])
	df["Total"]=df["Communication"]+df["Computation"]
	# os.makedirs("Results/SplitProfiling",exist_ok=True)
	# df.to_csv(f"Results/SplitProfiling/{worker}.csv")
	df=df[df["Total"]==df["Total"].min()]
	print(f"The optimal split point for worker {worker} is {df['Split Layer'].values[0]}")
	return df["Split Layer"].values[0]

def SplitPointProfileing(explist,splitpoints,subtractIdle=True,index=0):
	files=[]
	for expname in explist:
		files.append([file.split(".")[0][4:] for file in os.listdir(csvPath) if file.startswith("Log_" + expname)][0])
	EnergyInfo=[]
	for exp,f, s in zip(explist,files, splitpoints):
		# maxepoch=min(max(10, k * 3), 100)
		# print(f,s)
		df=CombineLogCSV(f, s)
		if subtractIdle:
			averageIdealpower = getAverageTimeFor(df, Message="Nothing")
		else:
			averageIdealpower = 0
		cummnication = getEnegryConsumption(df, contype="communication", basepower=averageIdealpower)[index]
		compution = getEnegryConsumption(df, contype="computation", basepower=averageIdealpower)[index]
		EnergyInfo.append((exp,s,cummnication,compution))
	df=pd.DataFrame(EnergyInfo,columns=["Experiment","Split Layer","Communication","Computation"])
	df["Total"]=df["Communication"]+df["Computation"]
	# os.makedirs("Results/SplitProfiling",exist_ok=True)
	print(df)
	return df

def decreaseNoiseLevel(initalclientstoNoise,subset,percent=100):
	fsimarray = pd.read_csv("Fsimmean.csv")
	allnoices=fsimarray.columns[1:].to_numpy().astype(float)
	maxnoise = initalclientstoNoise[np.max(subset)] * percent / 100
	decreasedNoise=np.max(allnoices[allnoices<=maxnoise])
	decreasedNoisedict={sp:decreasedNoise if sp in subset and n>decreasedNoise else n for sp, n in initalclientstoNoise.items()}
	return decreasedNoisedict


def getOptimimalSplitPoint(model,file,alpha,maxpoints,decreaseNoise=False,percent=100):
	noisesforSp={1: 1.3, 2: 1.3, 3: 0.25, 4: 0.35, 5: 0.75, 6: 0.3, 7: 0.6, 8: 0.4, 9: 0.15, 10: 0.05}
	fsimarray = pd.read_csv({"ResNet18":"Fsim/FsimmeanRestnet18.csv"}[model])
	fsimarray = fsimarray.set_index("SplitLayer")
	df=pd.read_csv(file)
	df=df[["Split Layer","Total"]]
	totalmax,totalmin=df["Total"].max(),df["Total"].min()
	fsimmax,fsimmin=fsimarray.max(),fsimarray.min()
	# Normalize energy and privacy leakage before the weighted P3SL objective.
	df["Total"]=(df["Total"]-totalmin)/(totalmax-totalmin) if totalmax != totalmin else 0
	fsimarray=(fsimarray-fsimmin)/(fsimmax-fsimmin)
	df=df[:maxpoints]
	energyconsumptins=[(i,df1["Split Layer"],df1["Total"]) for i,df1 in df.iterrows()]
	minimum=min(energyconsumptins,key=lambda x:x[2]) #This part is just remove the Split points before minimum power consumptions
	subsetenrgy=energyconsumptins[minimum[0]:]
	spsubset=[int(s[1]) for s in subsetenrgy]
	# print(spsubset)
	if decreaseNoise:
		noisesforSp= decreaseNoiseLevel(noisesforSp,subset=spsubset,percent=percent)
	ranklist=[]
	fsimarraydict= {sp:fsimarray.loc[int(sp), str(noise)] for sp,noise in noisesforSp.items()}
	for i,sp,engcons in subsetenrgy:
		ranklist.append((int(sp),alpha*fsimarraydict[int(sp)]+(1-alpha)*engcons))
	ranklist=sorted(ranklist,key=lambda x:x[1])
	sp=[s for s,r in ranklist][0]
	return sp,noisesforSp[sp]

def CalculateSplitPoints(model):
	# noisearray = pd.read_csv({"ResNet18":"Fsim/FsimmeanRestnet18.csv"}[model])
	folder={"ResNet18":"../ServerResults/FilteredResults/ResNet18/256/","VGG16bn":"../ServerResults/FilteredResults/Vgg16bn_Cifar10/"}[model]
	outfolder="../ServerResults/OptimalSP/"
	os.makedirs(outfolder,exist_ok=True)
	files=["alice.csv","alice1.csv","alice2.csv","alice3.csv"]
	# maxsplitpoints=[[10,4,2,2],[2,4,7,9]]
	# maxsplitpoints=[[10,4,7,8],[2,4,7,10]]
	# for msps in maxsplitpoints:
	for mspstr in os.listdir(folder):
	# 	mspstr="_".join([str(m) for m in msps])
		msps=[int(m) for m in mspstr.split("_")]
		mspfolder=folder+mspstr+"/"
		values = []
		for alpha in np.arange(0,1.05,0.1):
			values.append([getOptimimalSplitPoint(model,mspfolder+file,alpha=alpha,maxpoints=msp)[0] for file,msp in zip(files,msps)])
		df=pandas.DataFrame(values,columns=["Alice1","Alice2","Alice3","Alice4",])
		df["Alpha"]=[f"{s:.4}" for s in np.arange(0,1.05,0.1)]
		df.to_csv(outfolder+"OptimalPoints_"+model+"_"+mspstr+".csv",index=None)
		print("Result Saved at "+outfolder+"OptimalPoints_"+model+"_"+mspstr+".csv")



if __name__ == '__main__':
	print(calculatePowerConsumptions("local",24_10_25_15_40_41,pd.read_csv("CSV/KASALOGS/LOG_24_10_25_15_40_41.csv")))