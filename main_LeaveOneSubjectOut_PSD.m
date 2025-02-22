clear;clc;

addpath(genpath('.\Algorithm'))

subjects=32;
trials=40;
channels=40;
F=25;
dim_dr=8;
iter=200;
order=1;

frequency=2:2:50;
idx_delta=find(frequency>=1,1,'first'):find(frequency<=4,1,'last');
idx_theta=find(frequency>=4,1,'first'):find(frequency<=7,1,'last');
idx_alpha=find(frequency>=8,1,'first'):find(frequency<=13,1,'last');
idx_beta=find(frequency>=14,1,'first'):find(frequency<=30,1,'last');
idx_gamma=find(frequency>30,1,'first'):find(frequency<50,1,'last');
idx_fusion=find(frequency>=4,1,'first'):find(frequency<=45,1,'last');

acc_delta_dr=zeros(subjects,length(dim_dr));
acc_theta_dr=zeros(subjects,length(dim_dr));
acc_alpha_dr=zeros(subjects,length(dim_dr));
acc_beta_dr=zeros(subjects,length(dim_dr));
acc_gamma_dr=zeros(subjects,length(dim_dr));
acc_fusion_dr=zeros(subjects,length(dim_dr));

Label_cv_delta=zeros(length(dim_dr),trials,subjects);
Label_cv_theta=zeros(length(dim_dr),trials,subjects);
Label_cv_alpha=zeros(length(dim_dr),trials,subjects);
Label_cv_beta=zeros(length(dim_dr),trials,subjects);
Label_cv_gamma=zeros(length(dim_dr),trials,subjects);
Label_cv_fusion=zeros(length(dim_dr),trials,subjects);

PSD_gamma_NA_all=zeros(length(idx_gamma),channels,channels,subjects*trials);
PSD_beta_RA_all=zeros(length(idx_beta),channels,channels,subjects*trials);
PSD_fusion_NA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
PSD_fusion_RA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
Label_all=zeros(1,subjects*trials);
for i=1:subjects
    load(['..\Dataset\DEAP\DEAP_PSD\Data_S',num2str(i),'_PSD.mat']);

    % gamma band
    PSD_gamma_NA_all(:,:,:,(i-1)*trials+1:i*trials)=PSD_gamma;
    for f=1:length(idx_gamma)
        PSD_gamma_s=squeeze(PSD_gamma(f,:,:,:));
        PSD_center=sqrtm(riemann_mean(PSD_gamma_s));
        for t=1:trials
            PSD_gamma(f,:,:,t)=PSD_center\PSD_gamma_s(:,:,t)/PSD_center';
        end
    end
    PSD_gamma_RA_all(:,:,:,(i-1)*trials+1:i*trials)=PSD_gamma;

    % beta band
    PSD_beta_NA_all(:,:,:,(i-1)*trials+1:i*trials)=PSD_beta;
    for f=1:length(idx_beta)
        PSD_beta_s=squeeze(PSD_beta(f,:,:,:));
        PSD_center=sqrtm(riemann_mean(PSD_beta_s));
        for t=1:trials
            PSD_beta(f,:,:,t)=PSD_center\PSD_beta_s(:,:,t)/PSD_center';
        end
    end
    PSD_beta_RA_all(:,:,:,(i-1)*trials+1:i*trials)=PSD_beta;

    % fusion
    PSD_fusion_NA_all(:,:,:,(i-1)*trials+1:i*trials)=psd;

    PSD_fusion=psd;
    for f=1:length(idx_fusion)
        PSD_fusion_s=squeeze(PSD_fusion(f,:,:,:));
        PSD_center=sqrtm(riemann_mean(PSD_fusion_s));
        for t=1:trials
            PSD_fusion(f,:,:,t)=PSD_center\PSD_fusion_s(:,:,t)/PSD_center;
        end
    end
    PSD_fusion_RA_all(:,:,:,(i-1)*trials+1:i*trials)=PSD_fusion;
    
    Label_all((i-1)*trials+1:i*trials)=Label_valance;
end

for d=1:length(dim_dr)
    for s=1:subjects
        [acc_gamma_dr(s,d),Label_cv_gamma(d,:,s),W_gamma]=LeaveOneSubjectOutCV_PSD(s,order,iter,dim_dr(d),PSD_gamma_RA_all,Label_all);
    end
end

