clc
clear all
close all
addpath('datasets');
addpath('fun');
%% 更换数据集
data='ORL';      
%%
load(data);
for i=1:4
    new_fea{i}=double(fea{i});
end
fea=new_fea;
[~,v]=size(fea);
[n,~]=size(fea{1});
c=size(unique(gt),1);
class_num=c;
label=gt;
dimension=0;
for num = 1:v
    dimension=dimension+size(fea{num},2);
end
feanum=[10:10:300];
ACC=[];
NMI=[];
ACCstd=[];
NMIstd=[];
Fscore=[];
Fscorestd=[];
Precision=[];
Precisionstd=[];
Recall=[];
Recallstd=[];
ARI=[];
ARIstd=[];
rankmvufs=[];
mvufsobj=[];
%% 参数取值范围
xd_alpha=[1e-2,1e-1,1,1e1,1e2];
xd_beta=[1e-2,1e-1,1,1e1,1e2];
xd_deta=[1e-2,1e-1,1,1e1,1e2];
xd_deta_H=[1,1e1,1e2,1e3,1e4];
omega = [50,30,50,50];
%% 
c = length(unique(gt));
iter=0;
for xdi=1:size(xd_alpha,2)                  
    for xdo=1:size(xd_beta,2)             
        for xdz=1:size(xd_deta,2)    
            for xdh=1:size(xd_deta_H,2)    
                iter=iter+1     
                [index] = TDPSL(fea,xd_alpha(xdi),xd_beta(xdo),xd_deta(xdz),xd_deta_H(xdh),c,omega);
                XX = [];
                for i = 1:v
                    X{1,i} = ( fea{1,i}-repmat(min(fea{1,i}),n,1) ) ./repmat( max(fea{1,i})-min(fea{1,i}),n,1);
                    X{1,i}( isnan(X{1,i}) ) = 1;
                    XX = [XX X{1,i}];
                end
               
                for j = 1:length(feanum) 
                    acc=[];
                    nmi=[];
                    fscore=[];
                    precision=[];
                    recall=[];
                    ari=[];
                    for k = 1:30        
                        new_fea = XX(:,index(1:feanum(j)));  
                        idx = kmeans(new_fea, class_num,'MaxIter',200);
                        res = bestMap(label,idx);
                        
                        acc111 = length(find(label == res))/length(label); 
                        nmi111 = MutualInfo(label,idx); 
                        [f111,p111,r111] = compute_f(label,idx);
                        [ar111,~,~,~]=RandIndex(label,idx);
                        
                        acc=[acc;acc111];
                        nmi=[nmi;nmi111];
                        fscore=[fscore;f111];
                        precision=[precision;p111];
                        recall=[recall;r111];
                        ari=[ari;ar111];
                    end
                    ACC=[ACC;sum(acc)/30];
                    ACCstd=[ACCstd;std(acc)];
                    NMI=[NMI;sum(nmi)/30];
                    NMIstd=[NMIstd;std(nmi)];
                    ARI=[ARI;sum(ari)/30];
                    ARIstd=[ARIstd;std(ari)];
                    Fscore=[Fscore;sum(fscore)/30];
                    Fscorestd=[Fscorestd;std(fscore)];
                    Precision=[Precision;sum(precision)/30];
                    Precisionstd=[Precisionstd;std(precision)];
                end
            end
        end
    end
end

number=size(ACC,1);     
final=zeros(number,4);
for j=1:number          
    final(j,1:10)=[ACC(j,1),ACCstd(j,1),NMI(j,1),NMIstd(j,1),ARI(j,1),ARIstd(j,1),Fscore(j,1),Fscorestd(j,1),Precision(j,1),Precisionstd(j,1)];
end
[newvalue,endindex]=sort(ACC,'descend');
ORL_top_5acc=final(endindex(1:5),:);      

