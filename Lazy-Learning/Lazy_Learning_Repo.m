% NIR Simulation for Suncor/Mulang's data
% Alireza Kheradmand
%__________________________________________________________________________

clear all

clc

Str2=load('C:\Users\kheradma\Desktop\Materials on LW models\Simulation\VIS 40 dataset\v40 (1).mat');
% Str=load('C:\Users\kheradma\Desktop\Materials on LW models\Simulation\missing input\wierd.mat');
Data=Str2;
misss=load('G:\My Drive\Materials on LW models\Simulation\missing input\missing samples.mat');
mj=misss.mj;
% [~,e]=size(Data);
Xc=Str2.X;
Xd=Xc;
Y=Str2.y;
[n,m]=size(Xc);
[~,a]=size(Y);

for i=1:n
    mj(i,:)=sort(mj(i,:));
    for j=1:2
        Xc(i,mj(i,j))=NaN;
    end
end
X=Xc;
for j=1:n
    for i=1:m
        if isnan(X(j,i))
            if j==1
                X(j,i)=X(j+1,i);
            elseif j==n
                X(j,i)=X(j-1,i);
            else
                X(j,i)=0.5*(X(j-1,i)+X(j+1,i));
            end
        end
        
    end
end
% X=Xd;

for i=1:a
    for j=1:n
        if ~isnan(Y(1,i)) || ~isnan(Y(j,i))
            if isnan(Y(j,i))
                Y(j,i)=Y(j-1,i);
            end 
        else
            Y(j,i)=0;
        end
    end

end
X=X-mean(X);


S=std(X);
[n,m]=size(X);
[~,a]=size(Y);

for i=1:m
%     X(:,i)=X(:,i)/S(i);
    if(isnan(X(:,i)))
        X(:,i)=0;
    end
    
end



%% Locally weighted PLS
% for qu=3:n
%         Xw=X(1:qu-1,:);
%         Yw=Y(1:qu-1,:);
% 
%     xq=X(qu,:); yq=Y(qu,:);
%     
%     for j=1:qu-1
%         d(j)=sqrt((xq-Xw(j,:))*(xq-Xw(j,:))');
%     end
%     dr=d;
%     D=max(d);
%     for j=1:qu-1
% 
%         d(j)=d(j)/D;
%         W1(j,j)=sqrt(exp(-(d(j)^2)));
% %         W(j,j)=1/(d(j));
%     end
%     xbar1=0;ybar1=0;
% 
%             for i=1:qu-1
%                 xbar1=xbar1+W1(i,i)*Xw(i,:);
%                 ybar1=ybar1+W1(i,i)*Yw(i,:);
%             end
%             xbar1=xbar1/sum(diag(W1));
%             ybar1=ybar1/sum(diag(W1));
%             
%             Xw(:,1:m)=Xw(:,1:m)-xbar1(1:m);
%             Yw(:,1:a)=Yw(:,1:a)-ybar1(1:a);
%             A=xq;
%             xq=xq-xbar1;
%             yqhat1=ybar1;
%             A=xq;
%             for g=1:a
%                 for k=1:12
%                     u=Xw'*W1*Yw(:,g);
%                     t=Xw*u;
%                     p=t'*W1*Xw*inv(t'*W1*t);
%                     q=t'*W1*Yw(:,g)*inv(t'*W1*t);
%                     
%                     
%                     Xw=Xw-t*p;
%                     Yw(:,g)=Yw(:,g)-t*q;
%                     t=xq*u;
%                     xq=xq-t*p;
%                     yqhat1=yqhat1+t*q;
%                 end
%             end
%     Yest1(qu)=yqhat1;
% end
% Yest1=Yest1';
% Yest1(1:3)=Y(1:3);
% % Yest1(1:2)=0;
% MSE1=(Yest1-Y)'*(Yest1-Y)/n;


%% Trend-Based similarity as Weight + PLS
wsize=5;
for qu=wsize+1:n
t=0;q=0;p=0;u=0;
     xq=X(qu,:); yq=Y(qu,:);
     Xq=X(qu-wsize+1:qu,:);
%     Yq=[Y(qu-wsize+2:qu,:);yq];
        Xw=X(1:qu-1,:);
        Yw=Y(1:qu-1,:);
        lhanda=0.01;
        for i=1:qu-wsize
            cost(i)=0;
            Te=X(i:i+wsize-1,:);
            h=1;Q=0;T2=0;
            Xn=[Te;Xq];
            [XnLoadings,XnScores,~,~,varexp] = pca(Xn);
            for l=1:m
                if cumsum(varexp(1:l))>0.9
                    L=l;
                    break
                end
            end
            L=1;
            for ii=1:m
                Q=Q+(Xn(:,ii)-XnScores(:,1:L)*XnLoadings(ii,1:L)')'*(Xn(:,ii)-XnScores(:,1:L)*XnLoadings(ii,1:L)');
            end
            for jj=1:L
                T2=T2+XnScores(:,jj)'*XnScores(:,jj)/var(XnScores(:,jj));
            end
            cost(i)=lhanda*T2+(1-lhanda)*Q;
            
            
        end
    MM=max(cost);
    MN=min(cost);
    W2=0;
    for j=1:qu-wsize
        cost(j)=cost(j)/MM;
        r=cost(j);               
%         W2(j,j)=sqrt(exp(-(r^2)));
        W2(j,j)=1/(1+r);
%         W2(j,j)=1/cost(j);
%           W2(j,j)=(cost(j)-MN)/(MM-MN);
    end
    d1=dr(1:wsize);
    md1=max(d1);
    for j=1:wsize-1
        d(j)=d1(j)/md1;
%         Oldw(j,j)=sqrt(exp(-(d1(j)^2)));
        Oldw(j,j)=1/(1+d1(j));
    end
   W2=[Oldw zeros(length(Oldw), length(W2));zeros(length(W2), length(Oldw)) W2];
     xbar=0;ybar=0;

     for i=1:qu-1
         xbar=xbar+W2(i,i)*Xw(i,:);
         ybar=ybar+W2(i,i)*Yw(i,:);
     end
            xbar=xbar/sum(diag(W2));
            ybar=ybar/sum(diag(W2));
            
            Xw(:,1:m)=Xw(:,1:m)-xbar(1:m);
            Yw(:,1:a)=Yw(:,1:a)-ybar(1:a);
            
            xq=xq-xbar;
            yqhat2=ybar;
            
            for g=1:a
                for k=1:6
                    u=Xw'*W2*Yw(:,g);
                    t=Xw*u;
                    p=t'*W2*Xw*inv(t'*W2*t);
                    q=t'*W2*Yw(:,g)*inv(t'*W2*t);
                    
                    
                    Xw=Xw-t*p;
                    Yw(:,g)=Yw(:,g)-t*q;
                    t=xq*u;
                    xq=xq-t*p;
                    yqhat2=yqhat2+t*q;
                end
            end
Yest2(qu)=yqhat2;
e2=yq-yqhat2;

RMSE2(qu)=sqrt(e2*e2'/a);

end
Yest2(1:wsize)=Y(1:wsize);
Yest2=Yest2';
MSE2=(Yest2-Y)'*(Yest2-Y)/n;
R2=Rsquared(Y,Yest2);

labels{1}='Trad.JIT';labels{2}='Trend-based JIT';
figure, bar([MSE1 MSE2]')
xticklabels(labels)
ylabel('MSE')

figure, plot(Y/max(Y),'r'),hold on, plot(Yest1/max(Yest1),'b'), hold on, plot(Yest2/max(Yest2),'k')
xlabel('Time','FontWeight','bold')
ylabel('Normalized viscosity','FontWeight','bold')
legend('True value','Traditional JIT','TB-JIT')

figure, plot(0:1,0:1,'r'), hold on, scatter(Y/max(Y),Yest1/max(Yest1),'o'), hold on, scatter(Y/max(Y),Yest2/max(Yest2),'*')
xlabel('True value','FontWeight','bold'),ylabel('Prediction','FontWeight','bold')
legend('Reference Line','Traditional JIT','TB-JIT')

x_values=0:0.001:2;
pd1 = fitdist(Yest1/max(Yest1),'Normal');
pd2 = fitdist(Yest2/max(Yest2),'Normal');
pd = fitdist(Y/max(Y),'Normal');

yp1 = pdf(pd1,x_values);
yp2 = pdf(pd2,x_values);
yp = pdf(pd,x_values);

figure,
plot(x_values,yp1,'LineWidth',2), hold on, plot(x_values,yp2,'LineWidth',2)
hold on, plot(x_values,yp,'LineWidth',2)
grid on
legend('Traditional JIT','TB-JIT','Measurement')
xlabel('Normalizd Viscosity','FontWeight','bold')
ylabel('Probability Distribution Function','FontWeight','bold')

figure, plot(diag(W1)), hold on, plot(diag(W2))
legend('Traditional JIT','TB-JIT')
xlabel('Sample number','FontWeight','bold')
ylabel('Weight assigned','FontWeight','bold')
title('Weight assigned for sample no. 395','FontWeight','bold')
grid on

