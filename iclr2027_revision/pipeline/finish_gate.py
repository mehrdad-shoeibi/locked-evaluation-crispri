import time, os, sys, json, numpy as np
sys.path.insert(0,"iclr2027_revision/pipeline")
# wait for the running gate to finish
import subprocess
while subprocess.run(["pgrep","-f","run_repro_gate.py"],capture_output=True).stdout.strip():
    time.sleep(15)
R = json.load(open("iclr2027_revision/phase1/repro_gate.json"))
# corrected MAG-LINEAR: sklearn LinearRegression on 4-D m̃ (manuscript spec, phase11_strong_baselines:339)
import phase1_harness as H
from sklearn.linear_model import LinearRegression
base=H.load_base_shards(); tr,te=base["train"],base["test"]
y_tr,y_te=H.concat_y(tr),H.concat_y(te)
stats=H.fit_mtilde_stats(tr)
mt_tr=np.concatenate(H.mtilde(tr,stats),0); mt_te=np.concatenate(H.mtilde(te,stats),0)
lr=LinearRegression().fit(mt_tr,y_tr); mag=H.r2_pooled(y_te,lr.predict(mt_te))
R["MAG_LINEAR"]={"test_r2":mag,"manuscript":0.208,"delta":abs(mag-0.208),"pass":abs(mag-0.208)<=0.005,"model":"sklearn LinearRegression (manuscript spec)"}
R["GATE"]=all(R[k]["pass"] for k in ["MAG_LINEAR","RF_X_MTILDE","SPSM","MAG_SPSM"])
json.dump(R,open("iclr2027_revision/phase1/repro_gate_final.json","w"),indent=2,default=str)
print("MAG_LINEAR_FIXED",json.dumps(R["MAG_LINEAR"]))
print("GATE_FINAL",json.dumps({k:R[k] for k in ["MAG_LINEAR","RF_X_MTILDE","SPSM","MAG_SPSM","GATE"]},default=str))
