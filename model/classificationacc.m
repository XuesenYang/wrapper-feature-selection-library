function [fitness]=classificationacc(data,class,group,r,x,smile_subsample_segments,data_group,sa,n_d)
psoOptions.Obj.ub=size(data,2)-1;
popu111=floor(x);
changeRows1 = popu111<=0;
popu111(changeRows1)=1;
changeRows2 =popu111>psoOptions.Obj.ub;
popu111(changeRows2)=psoOptions.Obj.ub;
popu1=unique(popu111);
popu1length=length(popu1);
popu2=setdiff(1:psoOptions.Obj.ub,popu1);
randIndex = randperm(size(popu2,2));
popu3=popu2(1,randIndex);
val=[popu1 popu3(1,1:n_d-popu1length)];
for i=1:r-1    
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; %训练数据
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];%训练数据
    end
    mdl = fitcknn(data_tr(:,val),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%训练KNN
    Ac1=predict(mdl,data_ts(:,val)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    fitness=mean(Fit); %得到S个适应度值
end
