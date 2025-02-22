function [acc,label_cv,W]=LeaveOneTrialOutCV_PSD(psd,Label,dim,iter,order)
    % theta 2  alpha 3  beta 9  gamma 9
    F=size(psd,1);
    channels=size(psd,2);
    trials=length(Label);
    label_cv=zeros(size(Label));
    
    for i=1:trials
        idx_test=i;
        idx_train=1:trials;
        idx_train(i)=[];

        % prior_f=[0.15*ones(1,2),0.15*ones(1,3),0.35*ones(1,9),0.35*ones(1,9)];
        prior_f=[0.35*ones(1,9),0.65*ones(1,9)];
        W=zeros(channels,dim,F);
        label_f_test=zeros(1,F);
        label_f_train=zeros(1,length(idx_train));
        acc_f_train=zeros(1,F);
        for f=1:F
            data_test=squeeze(psd(f,:,:,idx_test));
            label_test=Label(idx_test);
            data_train=squeeze(psd(f,:,:,idx_train));
            label_train=Label(idx_train);
            
            W(:,:,f)=DR_PSD(data_train,label_train,dim,iter);

            for j=1:length(idx_train)
                idx_f_test=j;
                idx_f_train=1:length(idx_train);
                idx_f_train(idx_f_test)=[];
                label_f_train(j)=KNN(dim,W(:,:,f),data_train(:,:,idx_f_test),data_train(:,:,idx_f_train),label_train(idx_f_train),order);
            end
            acc_f_train(f)=mean(label_f_train'==label_train);

            label_f_test(f)=KNN(dim,W(:,:,f),data_test,data_train,label_train,order);
        end
        acc_f=prior_f.*acc_f_train/sum(prior_f.*acc_f_train);
        thre=sum(label_f_test.*acc_f);
        if(thre<=0.5)
            label_cv(i)=0;
        else
            label_cv(i)=1;
        end
    end
    
    acc=mean(label_cv==Label);

end
