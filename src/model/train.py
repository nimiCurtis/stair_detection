# system
import os
import hydra
from omegaconf import OmegaConf, DictConfig

# troch and ultralytics
from ultralytics import YOLO, settings
import torch
torch.cuda.empty_cache()

@hydra.main( version_base=None ,config_path="../../config/", config_name = "stair_detection")
def main(cfg:DictConfig):

    # get config and convert to dictionary
    model_cfg = OmegaConf.to_container(cfg.model,resolve=True)
    train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
    data_cfg = OmegaConf.to_container(cfg.data,resolve=True)

    # build data path based on config
    data = os.path.join(os.getcwd(),
                        data_cfg["root"],
                        data_cfg["project"]+'.'+data_cfg["version"]+'.'+data_cfg["model"],
                        'data.yaml')
    
    # Load a pretrained YOLO model using the model config
    model = YOLO(**model_cfg) 
    
    # Train the model using the train config and the data file path
    model.train(data=data,**train_cfg) 
    
    # change name of weights file
    experiment_dir = os.path.join(train_cfg["project"],
                                  train_cfg["name"],
                                  'weights')
    os.rename(src=os.path.join(experiment_dir,'best.pt'),
            dst=os.path.join(experiment_dir,train_cfg["name"]+'_best.pt'))

if __name__=='__main__':
    main()