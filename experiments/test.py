import argparse,sys,yaml,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import DCSS
from models.fault_score import InnovationScore,RobustNormalizer
from utils.data_loader import load_npz
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--data',required=True);p.add_argument('--checkpoint',default='checkpoints/best.pt');p.add_argument('--output',default='results/test_scores.npz');a=p.parse_args();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);c=state['config'];raw=load_npz(a.data);scale=lambda x:(x-state['scaler_mean'])/state['scaler_scale'];train=scale(raw['train'][:int(len(raw['train'])*c['data']['train_ratio'])]);val=scale(raw['train'][int(len(raw['train'])*c['data']['train_ratio']):]);test=scale(raw['test']);model=DCSS(state['input_dim'],c);model.load_state_dict(state['model']);model.eval();dep=state['dependency'];lag=c['dependency']['max_lag']
 def infer(x):
  ids=dep.assign(x);msg=torch.tensor(dep.messages(x,ids)[None],dtype=torch.float32);xx=torch.tensor(x[None],dtype=torch.float32)
  with torch.no_grad():o=model(xx,msg)
  return o['posterior'][0].numpy(),o['prior'][0].numpy()
 trp,trr=infer(train);vap,var=infer(val);tep,ter=infer(test);sc=InnovationScore(c['scoring']['covariance_regularization']).fit(trp,trr);norm=RobustNormalizer().fit(sc.score(trp,trr));vs=norm.transform(sc.score(vap,var));ts=norm.transform(sc.score(tep,ter));tau=np.quantile(vs,c['scoring']['threshold_quantile']);Path(a.output).parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,scores=ts,threshold=tau,offset=lag)
if __name__=='__main__':main()

