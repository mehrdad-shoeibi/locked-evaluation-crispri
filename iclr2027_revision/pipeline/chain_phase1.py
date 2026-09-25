import time, os, sys, json, subprocess
OUT=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "iclr2027_revision/phase1"))
# wait for gate to finish
while subprocess.run(["pgrep","-f","run_gate_final.py"],capture_output=True).stdout.strip():
    time.sleep(30)
if not os.path.exists(OUT+"/repro_gate_final.json"):
    open(OUT+"/chain_status.txt","w").write("GATE_CRASHED_NO_JSON"); raise SystemExit
R=json.load(open(OUT+"/repro_gate_final.json"))
if not R.get("GATE"):
    open(OUT+"/chain_status.txt","w").write("GATE_FAIL: "+json.dumps({k:R[k].get('pass') for k in ['MAG_LINEAR','RF_X_MTILDE','SPSM','MAG_SPSM']})); raise SystemExit
open(OUT+"/chain_status.txt","w").write("GATE_PASS -> launching Part C at "+time.strftime("%H:%M:%S"))
# launch Part C (val -> freeze -> test), sequential; writes frozen_selection.json + test_table.json
subprocess.run([sys.executable,
                "iclr2027_revision/pipeline/run_phase1_select.py"],
               stdout=open(OUT+"/partC.log","w"), stderr=subprocess.STDOUT, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
open(OUT+"/chain_status.txt","a").write("\nPartC done at "+time.strftime("%H:%M:%S"))
