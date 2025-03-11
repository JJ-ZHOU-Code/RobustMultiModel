from glob import glob
import numpy as np
import torch

import os
from sksurv.metrics import concordance_index_censored

def train_loop_survival_coattn(epoch, model, loader, optimizer, n_classes, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.
    train_recon_loss = 0.
    train_encode_wsi_loss = 0.
    train_align_loss = 0.
    train_kl_wsi = 0.
    train_kl_omic = 0.
    train_kl_joint = 0.
    train_kl_omic_componet = 0.

    if epoch < args.warm_epoch: 
        stage = 'warmup' 
    else: 
        stage = 'jointly' 
    
    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):
        
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)
        
        global kld_value
        kld_value = args.annealing_agent()
        if args.generator:
            hazards, S, Y_hat, A, all_loss  = model(stage=stage, train=True, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        else:
            hazards, S, Y_hat, A  = model(label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        loss = survey_loss + loss_reg
        if args.generator:
            loss += all_loss['recon_loss'] + all_loss['encode_wsi_loss']*args.beta + all_loss['align_loss'] * args.alpha +\
                  (all_loss['kl_wsi'] + all_loss['kl_omic_componet'] + all_loss['kl_omic'] + all_loss['kl_joint'])*kld_value*0.1
            all_risk_wsi_scores[batch_idx] = all_loss['risk_wsi']

        risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time
        
        train_loss_surv += survey_loss.item()
        if args.generator:
            train_recon_loss += all_loss['recon_loss'].item()
            train_encode_wsi_loss += all_loss['encode_wsi_loss']
            train_align_loss += all_loss['align_loss']
            train_kl_wsi += all_loss['kl_wsi']
            train_kl_omic += all_loss['kl_omic'].item()
            train_kl_joint += all_loss['kl_joint']
            train_kl_omic_componet += all_loss['kl_omic_componet']

        train_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss.item(), label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / gc + loss_reg
        loss.backward()
        
        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()
            args.annealing_agent.step()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader) 
    if args.generator: 
        train_recon_loss /= len(loader) 
        train_encode_wsi_loss /= len(loader) 
        train_align_loss /= len(loader) 
        train_kl_wsi /= len(loader) 
        train_kl_omic /= len(loader) 
        train_kl_joint /= len(loader) 
        train_kl_omic_componet /= len(loader)

    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_wsi_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_wsi_scores, tied_tol=1e-08)[0]
    if args.generator:
        train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_recon_loss: {:.4f}, train_encode_wsi_loss: {:.4f}, train_align_loss: {:.4f}, train_kl_wsi: {:.4f}, train_kl_omic: {:.4f}, train_kl_omic_componet: {:.4f}, train_kl_joint: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
                                epoch, train_loss_surv, train_recon_loss, train_encode_wsi_loss, train_align_loss, train_kl_wsi, train_kl_omic, train_kl_omic_componet, train_kl_joint, train_loss, c_index_train)
        acc_str = 'c_index_wsi_train: {:.4f}'.format(c_index_wsi_train) 
    else: 
        train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
            epoch, train_loss_surv, train_loss, c_index_train)
        acc_str = ''
    print(train_epoch_str)
    print(acc_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
        f.write(acc_str+'\n')
    f.close()

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)

        if args.generator:
            writer.add_scalar('train/recon_loss', train_recon_loss, epoch)
            writer.add_scalar('train/encode_wsi_loss', train_encode_wsi_loss, epoch)
            writer.add_scalar('train/align_loss', train_align_loss, epoch)
            writer.add_scalar('train/kl_wsi', train_kl_wsi, epoch)
            writer.add_scalar('train/kl_omic', train_kl_omic, epoch)
            writer.add_scalar('train/kl_joint', train_kl_joint, epoch)


def validate_survival_coattn(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_wsi_scores = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, all_loss  = model(train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                hazards, S, Y_hat, A  = model(label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        
        loss = survey_loss + loss_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += survey_loss.item()

        val_loss += loss.item()


    val_loss_surv /= len(loader)
    val_loss /= len(loader)

    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0.
        
    if args.generator:
        val_epoch_str = 'Epoch: {}, c_index: {:.4f}'.format(epoch, c_index)
    else: 
        val_epoch_str = 'Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, c_index: {:.4f}'.format(
            epoch, val_loss_surv, val_loss, c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return patient_results, c_index, True

    return patient_results, c_index, False

def validate_survival_coattn_missing(cur, epoch, model, loader, n_classes, early_stopping=None, monitor_cindex=None, writer=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None, args=None):
    model.eval()
    val_loss_surv, val_loss = 0., 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        if torch.randn(1) < args.missing_rate:
            omic_missing = True
        else: 
            omic_missing = False 

        data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
        data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
        data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
        data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
        data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
        data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if args.generator:
                hazards, S, Y_hat, A, recon_loss  = model(omic_missing=omic_missing, train=False, label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                hazards, S, Y_hat, A  = model(label=label, c=c, x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        survey_loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        loss = survey_loss + loss_reg

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += survey_loss.item()
        val_loss += loss.item()

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        c_index = 0
        
    val_epoch_str = "missing setting, val c-index: {:.4f}".format(c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    if writer:
        writer.add_scalar('val_missing/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val_missing/loss', val_loss, epoch)
        writer.add_scalar('val_missing/c-index', c_index, epoch)

    return patient_results, c_index, False