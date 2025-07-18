clear;clc;

subjects=15;
sessions=3;
trials=10;
channels=62;
num_window=35;


acc=zeros(1,subjects*sessions);
for su=1:45
    load(['C:\Users\DELL\Documents\MATLAB\Mode\SEED_Coherence_window\Data_S',num2str(su),'_Coherence.mat']);

    % beta band
    % Coh_beta=squeeze(mean(Coh(idx_beta,:,:,:),1));
    % Coh_beta=squeeze(mean(Coh,1));
    Coh_beta=reshape(Coh,size(Coh,1),size(Coh,2),[]);

    Label_w=zeros(1,trials*num_window);
    for j=1:trials
        for w=1:num_window
            Label_w((j-1)*num_window+w)=Label(j);
        end
    end
    Label=Label_w;


    p=4;
    m_train=trials-1;
    m_test=1;
    n=channels;
    lambda=0.5;
    beta=0.01;
    W=Coh_beta;
    D=zeros(size(Coh_beta));
    for i=1:size(Coh_beta,3)
        D(:,:,i)=sum(Coh_beta(:,:,i),2).*eye(channels);
    end
    L=D-W;
    LP=zeros(size(L));
    for i=1:size(Coh_beta,3)
        LP(:,:,i)=beta*eye(channels)-L(:,:,i);
    end
    delta=zeros(n,p);
    for i=1:p
        delta(i,i)=1;
    end
    
    
    for s=1:trials
        idx_test=(s-1)*num_window+1:s*num_window;
        idx_train=1:trials*num_window;
        %idx_train(idx_test)=[];
        m_train=length(idx_train);
        m_test=length(idx_test);

        label_test=Label(idx_test);
        label_train=Label(idx_train);
        W_train=W(:,:,idx_train);
        W_test=W(:,:,idx_test);
        L_train=L(:,:,idx_train);
        L_test=L(:,:,idx_test);
        LP_train=LP(:,:,idx_train);
        LP_test=LP(:,:,idx_test);
    
        phi=cell(length(idx_train),1);
        for i=1:m_train
            [H,~]=eig(L_train(:,:,i));
            phi{i,1}=H(:,1:p);
        end
    
        phimean=cell(2,1);
        for l=[0,1]
            [B,~]=eig(sum(L_train(:,:,label_train==l),3)/size(L_train(:,:,label_train==l),3));
            phimean{l+1,1}=B(:,1:p);
        end
        
        for iter=1
            for i=1:m_train
                phi_current=phi{i,1};
                err1=1;
                for iter1=1
                    theta=LP_train(:,:,i)*phi_current+lambda*phimean{label_train(i)+1,1};
                    [U,~,V]=svd(theta);
                    phi_next=U*(-delta)*V';
                    err1=norm(phi_next-phi_current,'fro');
                    phi_current=0.5*phi_current+0.5*phi_next;
                end
                phi{i,1}=phi_current;
            end
            
            for l=[0,1]
                label_c_idx=find(label_train==l);
                phimean{l+1,1}=phi{label_c_idx(randi(size(L_train(:,:,label_train==l),3))),1};
            end
        
            for l=[0,1]
                if l==0
                    l_other=2;
                else
                    l_other=1;
                end
        
                phi_c=phi(label_train==l,1);
                
                grad=1;
                % while norm(grad,'fro')>0.1
%                 for iter2=1:5
%                     grad=0;
%                     for i=1:size(phi_c,1)
%                         grad=grad-real((phimean{l+1,1}*phi_c{i,1}'*phimean{l+1,1}-phi_c{i,1}-lambda*(phimean{l+1,1}*phimean{l_other,1}'*phimean{l+1,1}-phimean{l_other,1})));
%                     end
%                     [Q,R]=qr((eye(n)-phimean{l+1,1}*phimean{l+1,1}')*grad,0);
%                     A=phimean{l+1,1}'*grad;
%                     BC=real(expm([A,-R';R,zeros(p)])*[eye(p);zeros(p)]);  
%                     phimean{l+1,1}=phimean{l+1,1}*BC(1:p,:)+Q*BC(p+1:2*p,:);
%                     % norm(grad,'fro')
%                 end
                for i=1:size(phi_c,1)
                    phimean{l+1,1}=phimean{l+1,1}+phi_c{i}/size(phi_c,1);
                end
            end
        end
        
%         phi_test=cell(length(idx_test),1);
%         dist=zeros(length(idx_test),2);
%         for i=1:m_test
%             [H,~]=eig(L_test(:,:,i));
%             phi_test{i,1}=H(:,1:p);
%             dist(i,1)=p-trace(phi_test{i,1}'*phimean{1,1});
%             dist(i,2)=p-trace(phi_test{i,1}'*phimean{2,1});
%         end
%         [~,label_pred]=min(dist,[],2);
%         acc(su)=acc(su)+sum(label_test==(label_pred-1)')/length(idx_test)/trials;


        phi_lda_train=zeros(length(idx_train),p*channels);
        for i=1:m_train
            H=phi{i};
            phi_lda_train(i,:)=H(:);
        end

        phi_lda_test=zeros(length(idx_test),p*channels);
        for i=1:m_test
            [H,~]=eig(L_test(:,:,i));
            H=H(:,1:p);
            phi_lda_test(i,:)=H(:);
        end

        lda=fitcdiscr(phi_lda_train,label_train,'DiscrimType','linear');
        label_pred=predict(lda,phi_lda_test)';
        accuracy = sum(label_pred==label_test)/numel(label_test);
        acc(su)=acc(su)+sum(label_test==label_pred)/length(idx_test)/trials;
    end
end

acc_sort=sort(acc,'descend');
acc_mean=mean(acc_sort(1:10));

















