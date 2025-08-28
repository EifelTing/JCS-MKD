import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base import Distiller

class JCS_MKD(Distiller):

    def __init__(self, student, teacher_list, cfg):
        super(JCS_MKD, self).__init__(student, teacher_list)
        self.ce_loss_weight = cfg.JCS_MKD.LOSS.CE_WEIGHT
        self.CAT_loss_weight = cfg.JCS_MKD.LOSS.CAT_loss_weight
        self.onlyCAT = cfg.JCS_MKD.onlyCAT
        self.CAM_RESOLUTION = cfg.JCS_MKD.LOSS.CAM_RESOLUTION
        self.relu = nn.ReLU()
        
        self.IF_NORMALIZE = cfg.JCS_MKD.IF_NORMALIZE
        self.IF_BINARIZE = cfg.JCS_MKD.IF_BINARIZE
        
        self.IF_OnlyTransferPartialCAMs = cfg.JCS_MKD.IF_OnlyTransferPartialCAMs
        self.CAMs_Nums = cfg.JCS_MKD.CAMs_Nums
        # 0: select CAMs with top x predicted classes
        # 1: select CAMs with the lowest x predicted classes
        self.Strategy = cfg.JCS_MKD.Strategy
        for teacher in self.teacher_list:
            teacher.cuda()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.fc = nn.Linear(cfg.DATASET.RESERVED_CLASS_NUM, cfg.DATASET.RESERVED_CLASS_NUM)
        self.zipf = 1 / torch.arange(1, cfg.DATASET.RESERVED_CLASS_NUM + 1).cuda()
        self.alpha = 1.0
        self.beta = 0.1
        self.mu = 0.005
        self.uskd_weight = 0.08
        self.self_distillation = True

    def forward_train(self, image, target, **kwargs):
        # 输出学生logit和学生中间特征
        logits_student, feature_student = self.student(image)

        logits_teacher_list = []
        feature_teacher_list = []

        with torch.no_grad():
            for teacher in self.teacher_list:
                logits_teacher, feature_teacher = teacher(image)
                logits_teacher_list.append(logits_teacher)
                feature_teacher_list.append(feature_teacher)

        # 根据教师的输出计算权重
        criterion_cls_lc = nn.CrossEntropyLoss(reduction='none')
        loss_t_list = [criterion_cls_lc(logit_t, target) for logit_t in logits_teacher_list]
        loss_t = torch.stack(loss_t_list, dim=0)
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (len(self.teacher_list) - 1)

        CAT_loss_list = []
        stu = feature_student["feats"][-1]
        for i in range(len(self.teacher_list)):
            tea = feature_teacher_list[i]["feats"][-1]

            # perform binarization
            if self.IF_BINARIZE:
                n,c,h,w = tea.shape
                threshold = torch.norm(tea, dim=(2,3), keepdim=True, p=1)/(h*w)
                tea = tea - threshold
                tea = self.relu(tea).bool() * torch.ones_like(tea)


            # only transfer CAMs of certain classes
            if self.IF_OnlyTransferPartialCAMs:
                n,c,w,h = tea.shape
                with torch.no_grad():
                    if self.Strategy==0:
                        l = torch.sort(logits_teacher_list[i], descending=True)[0][:, self.CAMs_Nums-1].view(n,1)
                        mask = self.relu(logits_teacher_list[i]-l).bool()
                        mask = mask.unsqueeze(-1).reshape(n,c,1,1)
                    elif self.Strategy==1:
                        l = torch.sort(logits_teacher_list[i], descending=True)[0][:, 99-self.CAMs_Nums].view(n,1)
                        mask = self.relu(logits_teacher_list[i]-l).bool()
                        mask = ~mask.unsqueeze(-1).reshape(n,c,1,1)
                tea,stu = _mask(tea,stu,mask)
            CAT_loss_list.append(CAT_loss(stu, tea, self.CAM_RESOLUTION, self.IF_NORMALIZE))

        loss_div = torch.stack(CAT_loss_list, dim=0)
        bsz = loss_div.shape[1]
        CAT_loss_our = torch.mul(attention, loss_div).sum() / (1.0 *bsz* len(self.teacher_list))
        loss_feat = self.CAT_loss_weight * CAT_loss_our


        loss_uskd = 0.0
        if self.self_distillation:
            # 自蒸馏
            if len(target.size()) > 1:
                value, label = torch.sort(target, descending=True, dim=-1)
                value = value[:, :2]
                label = label[:, :2]
            else:
                label = target.view(len(target), 1)
                value = torch.ones_like(label)
            N, c = logits_student.shape

            # final logit
            s_i = F.softmax(logits_student, dim=1)
            # 取出logits_student的正确类的值
            s_t = torch.gather(s_i, 1, label)

            # soft target label
            p_t = s_t ** 2
            p_t = p_t + value - p_t.mean(0, True)
            p_t[value == 0] = 0
            p_t = p_t.detach()

            s_i = self.log_softmax(logits_student)
            s_t = torch.gather(s_i, 1, label)
            # 正确类的引导损失
            loss_t = - (p_t * s_t).sum(dim=1).mean()

            # weak supervision
            if len(target.size()) > 1:
                target_label = target * 0.9 + 0.1 * torch.ones_like(logits_student) / c
            else:
                target_label = torch.zeros_like(logits_student).scatter_(1, label, 0.9) + 0.1 * torch.ones_like(logits_student) / c

            # weak logit
            # stu_avg = stu.reshape(stu.size(0), -1)
            stu_avg = feature_student['pooled_feat']
            w_fc = self.fc(stu_avg)
            w_i = self.log_softmax(w_fc)
            loss_weak = - (self.mu * target_label * w_i).sum(dim=1).mean()

            # N*class
            w_i = F.softmax(w_fc, dim=1)
            w_t = torch.gather(w_i, 1, label)

            # rank
            rank = w_i / (1 - w_t.sum(1, True) + 1e-6) + s_i / (1 - s_t.sum(1, True) + 1e-6)

            # soft non-target labels
            _, rank = torch.sort(rank, descending=True, dim=-1)
            z_i = self.zipf.repeat(N, 1)
            ids_sort = torch.argsort(rank)
            z_i = torch.gather(z_i, dim=1, index=ids_sort)

            mask = torch.ones_like(logits_student).scatter_(1, label, 0).bool()

            logit_s = logits_student[mask].reshape(N, -1)
            ns_i = self.log_softmax(logit_s)

            nz_i = z_i[mask].reshape(N, -1)
            nz_i = nz_i / nz_i.sum(dim=1, keepdim=True)

            nz_i = nz_i.detach()
            loss_non = - (nz_i * ns_i).sum(dim=1).mean()
            loss_uskd = self.alpha * loss_t + self.beta * loss_non + loss_weak

        # print(loss_feat.item(),self.uskd_weight * loss_uskd.item())
        loss_feat += self.uskd_weight * loss_uskd
        if self.onlyCAT is False:
            # print(target)
            loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
            losses_dict = {
                "loss_CE": loss_ce,
                "loss_CAT": loss_feat,
            }
        else:
            losses_dict = {
                "loss_CAT": loss_feat,
            }

        return logits_student, losses_dict


def _Normalize(feat,IF_NORMALIZE):
    if IF_NORMALIZE:
        feat = F.normalize(feat,dim=(2,3))
    return feat

def CAT_loss(CAM_Student, CAM_Teacher, CAM_RESOLUTION, IF_NORMALIZE):   
    CAM_Student = F.adaptive_avg_pool2d(CAM_Student, (CAM_RESOLUTION, CAM_RESOLUTION))
    CAM_Teacher = F.adaptive_avg_pool2d(CAM_Teacher, (CAM_RESOLUTION, CAM_RESOLUTION))
    loss = F.mse_loss(_Normalize(CAM_Student, IF_NORMALIZE), _Normalize(CAM_Teacher, IF_NORMALIZE),reduction='none').mean((-1,-2,-3))
    return loss
    

def _mask(tea,stu,mask):
    n,c,w,h = tea.shape
    mid = torch.ones(n,c,w,h).cuda()
    mask_temp = mask.view(n,c,1,1)*mid.bool()
    t=torch.masked_select(tea, mask_temp)
    
    if (len(t))%(n*w*h)!=0:
        return tea, stu

    n,c,w_stu,h_stu = stu.shape
    mid = torch.ones(n,c,w_stu,h_stu).cuda()
    mask = mask.view(n,c,1,1)*mid.bool()
    stu=torch.masked_select(stu, mask)
    
    return t.view(n,-1,w,h), stu.view(n,-1,w_stu,h_stu)