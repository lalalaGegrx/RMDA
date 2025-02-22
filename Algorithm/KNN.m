function  label_test=KNN(dim,W,data_test,data_train,label_train,order)

    trials_test=size(data_test,3);
    trials_train=size(data_train,3);
    label_test=zeros(1,trials_test);

    data_train_dr=zeros(dim,dim,trials_train);
    for i=1:trials_train
        data_train_dr(:,:,i)=W'*data_train(:,:,i)*W;
    end
    data_test_dr=zeros(dim,dim,trials_test);
    for i=1:trials_test
        data_test_dr(:,:,i)=W'*data_test(:,:,i)*W;
    end
    
    M=1e6;
    for i=1:trials_test
        d=zeros(1,trials_train);
        for j=1:trials_train
            d(j)=distance_jeffreys(data_test_dr(:,:,i),data_train_dr(:,:,j));
            if real(d(j))<0||abs(imag(d(j)))>1
                d(j)=M;
            else
                d(j)=abs(real(d(j)));
            end
        end

        [~,idx_sort]=sort(d,'ascend');
        label_train_sort=label_train(idx_sort);
        label_test(i)=mode(label_train_sort(1:order));
    end

end
