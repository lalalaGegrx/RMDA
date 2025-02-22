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

Cov_gamma_NA_all=zeros(length(idx_gamma),channels,channels,subjects*trials);
Cov_beta_RA_all=zeros(length(idx_beta),channels,channels,subjects*trials);
Cov_fusion_NA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
Cov_fusion_RA_all=zeros(length(idx_fusion),channels,channels,subjects*trials);
Label_all=zeros(1,subjects*trials);
for i=1:subjects
    load(['..\Dataset\DEAP\DEAP_Covariance\Data_S',num2str(i),'_Covariance.mat']);

    % gamma band
    Cov_gamma_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov_gamma;
    for f=1:length(idx_gamma)
        Cov_gamma_s=squeeze(Cov_gamma(f,:,:,:));
        Cov_center=sqrtm(riemann_mean(Cov_gamma_s));
        for t=1:trials
            Cov_gamma(f,:,:,t)=Cov_center\Cov_gamma_s(:,:,t)/Cov_center';
        end
    end
    Cov_gamma_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov_gamma;

    % beta band
    Cov_beta_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov_beta;
    for f=1:length(idx_beta)
        Cov_beta_s=squeeze(Cov_beta(f,:,:,:));
        Cov_center=sqrtm(riemann_mean(Cov_beta_s));
        for t=1:trials
            Cov_beta(f,:,:,t)=Cov_center\Cov_beta_s(:,:,t)/Cov_center';
        end
    end
    Cov_beta_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov_beta;

    % fusion
    Cov_fusion_NA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov;

    Cov_fusion=Cov;
    for f=1:length(idx_fusion)
        Cov_fusion_s=squeeze(Cov_fusion(f,:,:,:));
        Cov_center=sqrtm(riemann_mean(Cov_fusion_s));
        for t=1:trials
            Cov_fusion(f,:,:,t)=Cov_center\Cov_fusion_s(:,:,t)/Cov_center;
        end
    end
    Cov_fusion_RA_all(:,:,:,(i-1)*trials+1:i*trials)=Cov_fusion;
    
    Label_all((i-1)*trials+1:i*trials)=Label_valance;
end

for d=1:length(dim_dr)
    for s=1:subjects
        [acc_gamma_dr(s,d),Label_cv_gamma(d,:,s),W_gamma]=LeaveOneSubjectOutCV_Covariance(s,order,iter,dim_dr(d),Cov_gamma_RA_all,Label_all);
    end
end

