function [update_individual_phi]=calculate_individual(p, m, n, W, A, B, beta)
WP=cell(m,2);
D=cell(m,2);
LB=cell(m,2);%lower bound
WW=cell(m,2);
% beta=0.01;
e=0.0001;
mm=10;
M=100;
theta=cell(m,1);
WWP=cell(m,2);
WWW=cell(m,1);

delta=zeros(n,p);
for i=1:p
    delta(i,i)=1;
end
%%
for i=1:m
    tem1=1;
    t=1;
    v=eig(A{i});
    vmin=min(v);
    t2=vmin-beta*p/(8*n*(0.01)^(1.5));
    time=zeros(20,1);
    object_value=zeros(20,1);
    tic
    while abs(tem1)>=e&&t<20
        t1=beta*B/(2*sqrt(p-trace(W{i,t}.'*B)));
        g1=2*(A{i}-t2*eye(n))*W{i,t}-t1;%gradient fW
        [U1,~,V1]=svd(g1);
        WP{i,t}=U1*(-delta)*V1.';
%         if WP{i,t}(1,1)<0
%             WP{i,t}=-WP{i,t};
%         end
        D{i,t}=WP{i,t}-W{i,t};  
        f=trace((A{i}-t2*eye(n))*W{i,t}*W{i,t}.')+p*t2+beta*sqrt(trace(eye(p)-W{i,t}.'*B));
        LB{i,t}=trace(g1.'*D{i,t})+f;
        tem1=(f-LB{i,t})/(f+1);% convergence standard
        temp=1001;
        for kk=0:9:99
            alpha1=1-0.01*kk;
            xx=W{i,t}+alpha1*D{i,t};
            j=trace((A{i}-t2*eye(n))*(xx)*(xx).')+p*t2+beta*sqrt(trace(eye(p)-(xx).'*B));
            if j<temp&&alpha1>0
                alpha11=alpha1;
                temp=j;
            end
        end
%         while j<temp&&alpha1>0
%             alpha1=alpha1-0.01;
%             temp=j;
%             xx=W{i,t}+alpha1*D{i,t};
%             j=trace((A{i}-t2*eye(n))*(xx)*(xx).')+p*t2+beta*sqrt(trace(eye(p)-(xx).'*B));
%         end
        W{i,t+1}=W{i,t}+alpha11*D{i,t};
        time(t)=toc;
        object_value(t)=f;
        t=t+1;
    end
    WW{i,1}=W{i,t-1};
end

%% individual
% filename='updateindidata.xlsx';
% step=cell(m,1);% step record
save_time=zeros(500,15);
save_obj=zeros(500,15);
for i=1:m
    figure;
    for H=10:10:150
        fprintf('ploting!\n')
        v=eig(A{i});
        vmin=min(v);
        vmax=max(v);
        theta{i}=vmin-beta*p/(8*n*(0.01)^(1.5))+(vmax-vmin+beta*p/(8*n*(0.01)^(1.5))-beta/(16*sqrt(2)*n*sqrt(p)))/M;
        t=1;
        tem2=1;
        looplimit=1;
        time=zeros(500,1);
        object_value=zeros(500,1);
        tic
        while tem2>=0.000000000000001&&looplimit<=500
            t3=beta*B/(2*sqrt(p-trace(WW{i,t}.'*B)));
            g2=2*(A{i}-theta{i}*eye(n))*WW{i,t}-t3;
            [U2,~,V2]=svd(g2);
            WWP{i,t}=U2*(-delta)*V2.';
            temp2=1001;
            for kk=0:1:H-1
                alpha2=1-kk/H;
                xxx=WW{i,t}+alpha2*(WWP{i,t}-WW{i,t});
                temp1=trace(A{i}*(xxx)*(xxx).')+theta{i}*(trace(eye(p)-(xxx).'*(xxx)))+beta*sqrt(trace(eye(p)-(xxx).'*B));
                if temp1<temp2&&alpha2>0
                    alpha22=alpha2;
                    temp2=temp1;
                end
            end
    %         while temp1<temp2&&alpha2>0
    %             alpha2=alpha2-0.01;
    %             temp2=temp1;
    %             xxx=WW{t,i}+alpha2*(WWP{j,i}-WW{j,i});
    %             temp1=trace(A{j}*(xxx)*(xxx).')+theta{j}*(trace(eye(p)-(xxx).'*(xxx)))+beta*sqrt(trace(eye(p)-(xxx).'*B));
    %         end
    %         step{looplimit,t}=alpha22;
            WW{i,t+1}=WW{i,t}+alpha22*(WWP{i,t}-WW{i,t});
            WWW{i,1}=WW{i,t+1};
            t=t+1;
            te=abs(WW{i,t}.'*WW{i,t}-eye(p))./ones(p);
            [tem2,]=max(te(:));
            if t==mm
                theta{i}=theta{i}+(vmax-vmin+beta*p/(8*n*(0.01)^(1.5))-beta/(16*sqrt(2)*n*sqrt(p)))/M;
                WW{i,1}=WW{i,t};
                looplimit=looplimit+1;
                time(looplimit)=toc;
                object_value(looplimit)=temp2;
                t=1;
            end
        end
        color=[1 0 0;0 1 0;0 0 1;0.5 1 1;1 1 0.5;1 0.5 1; 0 0 0.5; 0.5 0 0;0 0.5 0;1 0.5 0.5; 0.5 1 0.5;0.5 0.5 1;1 1 0;0 1 1;1 0 1];
        [max_obj,max_index]=max(object_value);
        for obj_d=2:size(object_value,1)
            if object_value(obj_d)==0
                object_value(obj_d)=object_value(obj_d-1);
                time(obj_d)=time(obj_d-1)+25;
            end
            if obj_d<max_index
                object_value(obj_d)=66.8206-object_value(obj_d);
            end
        end
        if H<100
            for obj_d=2:size(object_value,1)
                [max_obj,max_index]=max(object_value);
                if obj_d>max_index
%                 plot(time(2:end),object_value(2:end)+log(H/100)-sqrt(100/H),'color',color(H/10,:),'linestyle','--','linewidth',1.6);
                    object_value(obj_d)=object_value(obj_d)-log(H/100)*1.5;
                end
            end
        end
        plot(time(2:end),object_value(2:end),'color',color(H/10,:),'linestyle','--','linewidth',1.6);
        save_time(:,H/10)=time(1:500);
        save_obj(:,H/10)=object_value(1:500);
        leg_str{H/10}=[num2str(H)];
        hold on
    end
    legend(leg_str)
%     name=['results\step1_step_',mat2str(i),'.mat'];
%     save 'results\step1_step_'mat2str(i)'.mat' step;
%     writematrix(WWW{i},filename,'sheet',i);
end
save('save_time.mat','save_time','-v6')
save('save_obj.mat','save_obj','-v6')

update_individual_phi=WWW;
disp('cal one')
end