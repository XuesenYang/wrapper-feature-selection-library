% function best_fitness = bsofs(fun,n_p,n_d,n_c,rang_l,rang_r,max_iteration)
% fun = fitness_function
% n_p; population sizem
% n_d; number of dimension
% n_c: number of clusters
% rang_l; left boundary of the dynamic range
% rang_r; right boundary of the dynamic range
clc
name={'isolet','sonar','Hill_Valley_without_noise_Training','Epileptic Seizure Recognitio','redwine'...
     'whitewine'};
 addpath(genpath('dataset'));
num_dataset=length(name);
i=2; %:num_dataset   %选择数据集，可以手动改
dataset=name{i};
switch dataset    
    case 'isolet'
 load('isolet.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'sonar'
 load('sonar.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Hill_Valley_without_noise_Training'
 load('Hill.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Epileptic Seizure Recognitio'
 load('Epileptic Seizure Recognitio.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'redwine'
 load('redwine.mat')
    dn=1:1:10;
    dnsize=10;
    dnd=1;
     case 'whitewine'
 load('whitewine.mat')
    dn=1:1:10;
    dnsize=10;
    dnd=1;
end 
% X = data(:,1:2);
% labels = data(:,end);
% figure(1)
% gscatter(X(:,1), X(:,2), labels,'rgb','osd');
% xlabel('feature1');
% ylabel('feature2');
maxrun=30;
   Best_fit=zeros(1,dnsize);
for run=1:maxrun
   Best_fitness=zeros(1,dnsize);
for dnf=dn
if length(unique(data(:,end)))>30
    datalabel=data(:,1);
    datafeature=data(:,2:end);
    data=[datafeature,datalabel];
    group=data(:,end);
    class=unique(data(:,end));
else
    group=data(:,end);
    class=unique(data(:,end));
end
rang_l=1;
rang_r=size(data,2)-1;
if rang_r<20
    tag=3;
elseif rang_r>=20 && rang_r<=40
    tag=5;
else 
    tag=10;
end
n_d=dnf;
n_p=50;
k=10;
max_iteration=100;
vmax=0.1*rang_r;%最大速度
w_start = 0.9;   %Initial inertia weight's value
w_end = 0.4;       %Final inertia weight
popu = rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the population of individuals
VStep =rand(n_p,n_d);%初始化速度
% popu = cell2mat(struct2cell(load('Pop.mat')))';
n_iteration = 0; % current iteration number
% initialize cluster probability to be zeros
best_fitness = 1000000*ones(max_iteration,1);
fitness_popu = 1000000*ones(n_p,1);  % store fitness value for each individual
%**************************************************************************
%**************************************************************************
%% calculate fitness for each individual in the initialized population
fitness=zeros(1,n_p);
for idx=1:n_p
popu111=floor(popu(idx,:));
popu1=unique(popu111);
popu1length=length(popu1);
popu2=setdiff(1:rang_r,popu1);
randIndex = randperm(size(popu2,2));
popu3=popu2(1,randIndex);
val=[popu1 popu3(1,1:n_d-popu1length)];
for i=1:length(class)
    sa=[];
    sa=data((group==class(i)),:);
    [number_of_smile_samples,~] = size(sa); % Column-observation
    smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,k)); % indices of subsample segmentation points    
    data_group{i}=sa;
    smile_subsample_segments{i}=smile_subsample_segments1;
end
for i=1:k-1    
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
    fitness(1,idx)=mean(Fit);
end
    fitness_popu=fitness';
PBest = popu;
fPBest =fitness_popu;
[fGBest, g] = min(fPBest);
GBest = popu(g,:);
c1_now= 2.0;
c2_now= 2.0;
while n_iteration < max_iteration
         indi_temp=popu;
         w_now = ((w_start-w_end)*(max_iteration-n_iteration)/max_iteration)+w_end;
         R1 = rand(n_p,n_d);
         R2 = rand(n_p,n_d);
         A= repmat(indi_temp(g,:), n_p, 1);
         VStep =  w_now*VStep + c1_now*R1.*(PBest-indi_temp) + c2_now*R2.*(A-indi_temp);
         changeRows = VStep > vmax;
         VStep(find(changeRows)) =vmax;
         changeRows = VStep < -vmax;
        VStep(find(changeRows)) = -vmax;
        indi_temp=indi_temp+VStep;
        changeRow = indi_temp > rang_r;
        indi_temp(find(changeRow)) =rang_r;
        changeRows = indi_temp < rang_l;
        indi_temp(find(changeRow)) =rang_l;
  for idx=1:n_p
        indi_temp111=round(indi_temp(idx,:));
        changeRows1 = indi_temp111<=0;
        indi_temp111(changeRows1)=1;
        changeRows2 = indi_temp111>rang_r;
        indi_temp111(changeRows2)=rang_r;
        indi_temp1=unique(indi_temp111);
        indi_temp1length=length(indi_temp1);
        indi_temp2=setdiff(1:rang_r,indi_temp1);
        randIndex = randperm(size(indi_temp2,2));
        indi_temp3=indi_temp2(1,randIndex);
        val=[indi_temp1 indi_temp3(1,1:n_d-indi_temp1length)];
        for i=1:k-1    
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
        Fit_temp(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
        end
        fitness(1,idx)=mean(Fit_temp);
        fv = fitness(1,idx);
        if fv < fitness_popu(idx,1)  % better than the previous one, replace
            fitness_popu(idx,1) = fv;
            popu(idx,:) = indi_temp(idx,:);
        end        
  end
    n_iteration = n_iteration +1;
    PBest = popu;
    fPBest =fitness_popu;
    [fGBest, g] = min(fPBest);
    GBest = popu(g,:);
    % record the best fitness in each iteration
    fprintf('RUN: %d \t subsetsize: %d \t Iter: %d \t Err: %.4f \t \n',run,dnf,n_iteration,fGBest)
end
    Best_fitness(1,dnf/dnd)= fGBest;
end
    Best_fit(run,:)=Best_fitness;
end
    BESTFIT=(1-mean(Best_fit))*100;
    save('psofs','Best_fit','BESTFIT')
    
    

