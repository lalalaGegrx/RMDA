clear;clc;

subjects=32;
F=25;
interval=2;
for i=1:subjects
    disp(['Subject ',num2str(i)]);
    load(['..\Dataset\DEAP\Data_S',num2str(i),'.mat']);
    [trials,channels,T]=size(Data);

    Coh=zeros(F,channels,channels,trials);
    for trial=1:trials
        disp(['Trial ',num2str(trial)]);
        for j=1:channels
            for k=1:j
                [C,frequency]=mscohere(squeeze(Data(trial,j,:)),squeeze(Data(trial,k,:)),1000,200,1000,200);
                for f=1:F
                    idx=find(frequency>=interval*(f-1)+1,1,'first'):find(frequency<=interval*f+1,1,'last'); % interval=2
                    Coh(f,k,j,trial)=mean(C(idx));
                    Coh(f,j,k,trial)=mean(C(idx));
                end
            end
        end
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
    
    save(['..\Dataset\DEAP\DEAP_Coherence\Data_S',num2str(i),'_Coherence.mat'],'Coh','frequency','Label_valence','Label_arousal');

end
