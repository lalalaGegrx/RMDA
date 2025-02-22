clear;clc;

% F_delta=bandpass_filter_delta;
% F_theta=bandpass_filter_theta;
% F_alpha=bandpass_filter_alpha;
% F_beta=bandpass_filter_beta;
% F_gamma=bandpass_filter_gamma;

F_delta=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',1,'HalfPowerFrequency2',4, ...
         'SampleRate',200);
F_theta=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',4,'HalfPowerFrequency2',7, ...
         'SampleRate',200);
F_alpha=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',7,'HalfPowerFrequency2',13, ...
         'SampleRate',200);
F_beta=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',13,'HalfPowerFrequency2',30, ...
         'SampleRate',200);
F_gamma=designfilt('bandpassiir','FilterOrder',20, ...
         'HalfPowerFrequency1',31,'HalfPowerFrequency2',50, ...
         'SampleRate',200);

for i=1:45
    load(['Data_S',num2str(i),'.mat']);
    idx=find(Label~=0);
    Data=Data(idx,:,:);
    Label=Label(idx);
    [trials,channels,T]=size(Data);

    idx_1=find(Label==1);
    idx_2=find(Label==-1);
    Data_trans=zeros(size(Data));
    Data_trans(1:length(idx_2),:,:)=Data(idx_2,:,:);
    Data_trans(length(idx_2)+1:length(idx),:,:)=Data(idx_1,:,:);
    Label(1:length(idx_2))=0;
    Label(length(idx_2)+1:length(idx))=1;
    
    Cov_delta=zeros(channels,channels,trials);
    Cov_theta=zeros(channels,channels,trials);
    Cov_alpha=zeros(channels,channels,trials);
    Cov_beta=zeros(channels,channels,trials);
    Cov_gamma=zeros(channels,channels,trials);
    for j=1:trials
        data_delta=zeros(channels,T);
        data_theta=zeros(channels,T);
        data_alpha=zeros(channels,T);
        data_beta=zeros(channels,T);
        data_gamma=zeros(channels,T);
        for c=1:channels
            data_single=squeeze(Data_trans(j,c,:));
            data_delta(c,:)=filtfilt(F_delta,data_single);
            data_theta(c,:)=filtfilt(F_theta,data_single);
            data_alpha(c,:)=filtfilt(F_alpha,data_single);
            data_beta(c,:)=filtfilt(F_beta,data_single);
            data_gamma(c,:)=filtfilt(F_gamma,data_single);
        end
        Cov_delta(:,:,j)=covariances(data_delta);
        Cov_theta(:,:,j)=covariances(data_theta);
        Cov_alpha(:,:,j)=covariances(data_alpha);
        Cov_beta(:,:,j)=covariances(data_beta);
        Cov_gamma(:,:,j)=covariances(data_gamma);
    end
    
    % Label
    Label_v=labels(:,1);
    Label_a=labels(:,2);
    Label_valance=zeros(trials,1);
    Label_arousal=zeros(trials,1);
    Label=labels;
    for j=1:trials
        if(Label_v(j)<=5)
            Label_valance(j)=0;
        elseif(Label_v(j)>5)
            Label_valance(j)=1;
        end
        if(Label_a(j)<=5)
            Label_arousal(j)=0;
        elseif(Label_a(j)>5)
            Label_arousal(j)=1;
        end
    end

    save(['..\Dataset\DEAP\DEAP_Covariance\Data_S',num2str(i),'_Covariance.mat'],'Cov_delta','Cov_theta','Cov_alpha','Cov_beta','Cov_gamma','Label_valence','Label_arousal');

end
