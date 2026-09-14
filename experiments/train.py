import argparse,sys,yaml,numpy as np,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import DCSS
from models.dynamic_dependency import DynamicDependency
from utils.data_loader import load_npz,split_scale
from utils.seed import set_seed
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/default.yaml');p.add_argument('--data',required=True);p.add_argument('--output',default='checkpoints/best.pt');a=p.parse_args();c=yaml.safe_load(open(a.config));set_seed(c['seed']);raw=load_npz(a.data);train,val,scaler=split_scale(raw['train'],c['data']['train_ratio']);d=c['dependency'];dep=DynamicDependency(d['regimes'],d['max_lag'],d['top_k'],d['ridge_alpha'],d['mechanism_window'],d['mechanism_stride']).fit(train);model=DCSS(train.shape[1],c);opt=torch.optim.Adam(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']);best=float('inf')
 # Compact full-sequence windows; production users may replace with a streaming loader.
 def batch(x):
  ids=dep.assign(x);msg=dep.messages(x,ids);w=c['data']['window_length'];xs=[];ms=[]
  for i in range(len(x)-w+1): xs.append(x[i:i+w]);ms.append(msg[i:i+w-d['max_lag']])
  return torch.tensor(np.asarray(xs),dtype=torch.float32),torch.tensor(np.asarray(ms),dtype=torch.float32)
 xt,mt=batch(train);xv,mv=batch(val)
 for epoch in range(c['training']['epochs']):
  model.train(); order=torch.randperm(len(xt))
  for i in range(0,len(order),c['training']['batch_size']):
   ix=order[i:i+c['training']['batch_size']];o=model(xt[ix],mt[ix]);target=xt[ix,:, :];weights=c['loss'];loss=weights['reconstruction']*(o['reconstruction']-target).square().mean()+weights['prediction']*(o['prediction']-target[:,d['max_lag']:]).square().mean()+weights['latent']*(o['posterior']-o['encoded'][:,d['max_lag']:]).square().mean()+weights['predictive_nll']*o['nll'];opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip']);opt.step()
  model.eval()
  with torch.no_grad(): o=model(xv,mv);vl=(o['prediction']-xv[:,d['max_lag']:]).square().mean().item()
  if vl<best: best=vl;Path(a.output).parent.mkdir(parents=True,exist_ok=True);torch.save({'model':model.state_dict(),'config':c,'input_dim':train.shape[1],'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_,'dependency':dep},a.output)
if __name__=='__main__':main()
