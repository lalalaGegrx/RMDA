clear;clc;

addpath(genpath('.\Algorithm'))

subjects=32;
trials=40;
channels=32;
dim_dr=8;
iter=20;
order=1;

acc_theta=zeros(subjects,length(dim_dr));
acc_alpha=zeros(subjects,length(dim_dr));
acc_beta=zeros(subjects,length(dim_dr));
acc_gamma=zeros(subjects,length(dim_dr));
acc_fusion=zeros(subjects,length(dim_dr));
Label_gamma=zeros(trials,length(dim_dr),subjects);
Label_true=zeros(trials,subjects);

for i=1:subjects
    load(['..\Dataset\DEAP\DEAP_Coherence\Data_S',num2str(i),'_Coherence.mat']);

    Label_true(:,i)=Label_valance;
    % Label_true(:,i)=Label_arousal;
    for j=1:length(dim_dr)
        [acc_gamma(i,j),Label_gamma(:,j,i),W]=LeaveOneTrialOutCV_Coherence(dim_dr(j),Coh_gamma,Label_valance,order,iter);
        %[acc_gamma(i,j),Label_gamma(:,j,i),W]=LeaveOneTrialOutCV_Coherence(dim_dr(j),Coh_gamma,Label_arousal,order,iter);
    end
end

