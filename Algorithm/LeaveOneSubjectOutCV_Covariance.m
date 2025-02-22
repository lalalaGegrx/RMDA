function [acc,label_cv,W]=LeaveOneSubjectOutCV_Covariance(idx,dim,Cov,Label,iter,order)
    % beta:9  gamma:9
    subjects=32;
    sessions=1;
    trials=40;
    channels=40;
    F=size(Cov,1);

    idx_test=(idx-1)*trials*sessions+1:idx*trials*sessions;
    idx_train=1:subjects*trials*sessions;
    idx_train(idx_test)=[];
    label_test=Label(idx_test);
    label_train=Label(idx_train);

    label_cv=zeros(1,trials*sessions);
    prior_f=[0.6*ones(1,9),0.4*ones(1,9)];
    W=zeros(channels,dim,F);
    label_f_test=zeros(trials*sessions,F);
    label_f_train=zeros(1,length(idx_train));
    acc_f_train=zeros(1,F);
    for f=1:F
        data_test=squeeze(Cov(f,:,:,idx_test));
        data_train=squeeze(Cov(f,:,:,idx_train));
        
        W(:,:,f)=DR_Covariance(data_train,label_train,dim,iter);
        for j=1:length(idx_train)
            idx_f_test=j;
            idx_f_train=1:length(idx_train);
            idx_f_train(idx_f_test)=[];
            label_f_train(j)=KNN(dim,W(:,:,f),data_train(:,:,idx_f_test),data_train(:,:,idx_f_train),label_train(idx_f_train),order);
        end
        acc_f_train(f)=mean(label_f_train==label_train);

        label_f_test(:,f)=KNN(dim,W(:,:,f),data_test,data_train,label_train,order);
    end
    acc_f=prior_f.*acc_f_train/sum(prior_f.*acc_f_train);
    for i=1:trials*sessions
        thre=sum(label_f_test(i,:).*acc_f);
        if(thre<=0.5)
            label_cv(i)=0;
        else
            label_cv(i)=1;
        end
    end
    
    acc=mean(label_cv==label_test);

end
