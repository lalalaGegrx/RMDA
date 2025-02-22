clear;clc;

addpath(genpath('.\Algorithm'))

subjects=32;
trials=40;
channels=40;
F=25;
dim_dr=4;
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

Coh_gamma_NA_all=zeros(length(idx_gamma),channels,channels,subjects*trials);
Coh_beta_RA_all=zeros(length(idx_beta),channels,channels,subjects*trials);
Coh_fusion_NA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
Coh_fusion_RA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
Label_all=zeros(1,subjects*trials);
for i=1:subjects
    load(['..\Dataset\DEAP\DEAP_Coherence\Data_S',num2str(i),'_Coherence.mat']);

    % gamma band
    Coh_gamma_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh_gamma;
    for f=1:length(idx_gamma)
        Coh_gamma_s=squeeze(Coh_gamma(f,:,:,:));
        Coh_center=sqrtm(riemann_mean(Coh_gamma_s));
        for t=1:trials
            Coh_gamma(f,:,:,t)=Coh_center\Coh_gamma_s(:,:,t)/Coh_center';
        end
    end
    Coh_gamma_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh_gamma;

    % beta band
    Coh_beta_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh_beta;
    for f=1:length(idx_beta)
        Coh_beta_s=squeeze(Coh_beta(f,:,:,:));
        Coh_center=sqrtm(riemann_mean(Coh_beta_s));
        for t=1:trials
            Coh_beta(f,:,:,t)=Coh_center\Coh_beta_s(:,:,t)/Coh_center';
        end
    end
    Coh_beta_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh_beta;

    % fusion
    Coh_fusion_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh;

    Coh_fusion=Coh;
    for f=1:length(idx_fusion)
        Coh_fusion_s=squeeze(Coh_fusion(f,:,:,:));
        Coh_center=sqrtm(riemann_mean(Coh_fusion_s));
        for t=1:trials
            Coh_fusion(f,:,:,t)=Coh_center\Coh_fusion_s(:,:,t)/Coh_center;
        end
    end
    Coh_fusion_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Coh_fusion;
    
    Label_all((i-1)*trials+1:i*trials)=Label_valance;
end

for d=1:length(dim_dr)
    for s=1:subjects
        [acc_gamma_dr(s,d),Label_cv_gamma(d,:,s),W_gamma]=LeaveOneSubjectOutCV_Coherence(s,order,iter,dim_dr(d),Coh_gamma_RA_all,Label_all);
    end
end

