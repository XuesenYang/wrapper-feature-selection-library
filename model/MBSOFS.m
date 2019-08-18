% fun = fitness_function
% n_p; population size
% n_d; number of dimension
% n_c: number of clusters
% rang_l; left boundary of the dynamic range
% rang_r; right boundary of the dynamic range
% Modified brainstorming optimization algorithm for feature selection
clc
name={'isolet','sonar','Hill_Valley_without_noise_Training','Epileptic Seizure Recognitio','redwine'...
     'whitewine','MF','SPECTHeart','Statlog','Madelon','Libras Movement','LSVT_voice_rehabilitation','drivFaceD'...
     'Urban land cover','MEU-Mobile KSD 2016','ionosphere'};
 addpath(genpath('dataset'));
num_dataset=length(name);
i=2; %:num_dataset   
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
    case 'MF'
 load('MF.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'SPECTHeart'
 load('SPECTHeart.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
     case 'Statlog'
 load('Statlog.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
    case 'Madelon'
 load('Madelon.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'Libras Movement'
 load('Libras Movement.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'LSVT_voice_rehabilitation'
 load('LSVT_voice_rehabilitation.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'drivFaceD'
 load('drivFaceD.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5;
   case 'Urban land cover'
 load('Urban land cover.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5; 
    case 'MEU-Mobile KSD 2016'
 load('MEU-Mobile KSD 2016.mat')
    dn=5:5:50;
    dnsize=10;
    dnd=5; 
    case 'ionosphere'
 load('ionosphere.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
end         
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
n_c=5;
k=10;
max_iteration=100;
prob_one_cluster = 0.8; % probability for select one cluster to form new individual; 
stepSize = ones(1,n_d); % effecting the step size of generating new individuals by adding random values
popu = rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the population of individuals
% popu = cell2mat(struct2cell(load('Pop.mat')))';
popu_sorted  = rang_l + (rang_r - rang_l) * rand(n_p,n_d); % initialize the  population of individuals sorted according to clusters
n_iteration = 0; % current iteration number
% initialize cluster probability to be zeros
prob = zeros(n_c,1);
best = zeros(n_c,1);  % index of best individual in each cluster
centers = rang_l + (rang_r - rang_l) * rand(n_c,n_d);  % initialize best individual in each cluster
centers_copy = rang_l + (rang_r - rang_l) * rand(n_c,n_d);  % initialize best individual-COPY in each cluster FOR the purpose of introduce random best
best_fitness = 1000000*ones(max_iteration,1);
fitness_popu = 1000000*ones(n_p,1);  % store fitness value for each individual
fitness_popu_sorted = 1000000*ones(n_p,1);  % store  fitness value for each sorted individual
indi_temp = zeros(1,n_d);  % store temperary individual
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
      data_ts=[test;data_ts] ; %
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];%
    end
    mdl = fitcknn(data_tr(:,val),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%
    Ac1=predict(mdl,data_ts(:,val)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    fitness(1,idx)=mean(Fit);
end
    fitness_popu=fitness';
while n_iteration < max_iteration
    randIndex=randperm(n_p);
    randselect=randIndex(1:n_c);%
    restselect=setdiff(1:n_p,randselect);%
    dij=zeros(size(restselect,2),size(randselect,2));
    Belong=zeros(1,size(restselect,2));
    cluster=zeros(n_p,1);
    for it=1:size(restselect,2)
        for jt=1:size(randselect,2)
            dij(it,jt)=sqrt(sum((popu(restselect(it),:)-popu(randselect(jt),:)).^2));
            [bn,belong]=min(dij(it,:),[],2);%%%
        end
            Belong(1,it)=belong;%%%
    end
    cluster(restselect,1)=Belong;
    cluster(randselect,1)=[1:n_c]';
%     cluster = kmeans(popu, n_c,'Distance','cityblock','Start',centers,'EmptyAction','singleton') % k-mean cluster  
    % clustering    
    fit_values = 100000000000000000000000000.0*ones(n_c,1);  % assign a initial big fitness value  as best fitness for each cluster in minimization problems
    number_in_cluster = zeros(n_c,1);  % initialize 0 individual in each cluster      
    for idx = 1:n_p
        number_in_cluster(cluster(idx,1),1)= number_in_cluster(cluster(idx,1),1) + 1;      
        % find the best individual in each cluster
        if fit_values(cluster(idx,1),1) > fitness_popu(idx,1)  % minimization
            fit_values(cluster(idx,1),1) = fitness_popu(idx,1);
            best(cluster(idx,1),1) = idx;
        end            
    end  
    best;   
    % form population sorted according to clusters
    counter_cluster = zeros(n_c,1);  % initialize cluster counter to be 0     
    acculate_num_cluster = zeros(n_c,1);  % initialize accumulated number of individuals in previous clusters    
    for idx =2:n_c
        acculate_num_cluster(idx,1) = acculate_num_cluster((idx-1),1) + number_in_cluster((idx-1),1);%%%
    end    
    %start form sorted population
    for idx = 1:n_p
        counter_cluster(cluster(idx,1),1) = counter_cluster(cluster(idx,1),1) + 1 ;
        temIdx = acculate_num_cluster(cluster(idx,1),1) +  counter_cluster(cluster(idx,1),1);
        popu_sorted(temIdx,:) = popu(idx,:);
        fitness_popu_sorted(temIdx,1) = fitness_popu(idx,1);
    end       
    % record the best individual in each cluster
    for idx = 1:n_c
        centers(idx,:) = popu(best(idx,1),:);        
    end
    centers_copy = centers;  % make a copy
    
    if (rand() < 0.2) %  select one cluster center to be replaced by a randomly generated center
        cenIdx = ceil(rand()*n_c);
        centers(cenIdx,:) = rang_l + (rang_r - rang_l) * rand(1,n_d);
    end           
    % calculate cluster probabilities based on number of individuals in
    % each cluster
    for idx = 1:n_c
        prob(idx,1) = number_in_cluster(idx,1)/n_p;
        if idx > 1
            prob(idx,1) = prob(idx,1) + prob(idx-1,1);
        end
    end    
    % generate n_p new individuals by adding Gaussian random values                   
    for idx = 1:n_p
        r_1 = rand();  % probability for select one cluster to form new individual
        if r_1 < prob_one_cluster % select one cluster
            r = rand();
            for idj = 1:n_c
                if r < prob(idj,1)                      
                    if rand() < 0.4  % use the center
                       indi_temp(1,:) = centers(idj,:); 
                    else % use one randomly selected  cluster
                        indi_1 = acculate_num_cluster(idj,1) + ceil(rand() * number_in_cluster(idj,1));
                        indi_temp(1,:) = popu_sorted(indi_1,:);  
                    end
                    break
                end
            end
        else % select two clusters
            % pick two clusters 
            cluster_1 = ceil(rand() * n_c);
            indi_1 = acculate_num_cluster(cluster_1,1) + ceil(rand() * number_in_cluster(cluster_1,1));          
            cluster_2 = ceil(rand() * n_c);
            indi_2 = acculate_num_cluster(cluster_2,1) + ceil(rand() * number_in_cluster(cluster_2,1));         
            tem = rand();
            if rand() < 0.005 %use center
                indi_temp(1,:) = rand*(rang_r-rang_l)+rang_l;
            else   % use randomly selected individuals from each cluster            
                indi_temp(1,:) =popu(idx,:)+ tem * (popu_sorted(indi_1,:)- popu_sorted(indi_2,:)); 
            end
        end                 
        stepSize = logsig(((0.5*max_iteration - n_iteration)/20)) * rand(1,n_d);
        indi_temp(1,:) = indi_temp(1,:) + stepSize .* normrnd(0,1,1,n_d);
        % if better than the previous one, replace it
        
        indi_temp111=round(indi_temp);
        changeRows1 = find(indi_temp111<=0);
        indi_temp111(changeRows1)=1;
        changeRows2 = find(indi_temp111>rang_r);
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
      data_ts=[test;data_ts] ; %
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];
    end
        mdl = fitcknn(data_tr(:,val),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%
        Ac1=predict(mdl,data_ts(:,val)); 
        Fit_temp(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
            fitness(1,idx)=mean(Fit_temp);
            fv = fitness(1,idx);
        if fv < fitness_popu(idx,1)  % better than the previous one, replace
            fitness_popu(idx,1) = fv;
            popu(idx,:) = indi_temp(1,:);
        end         
    end        
    % keep the best for each cluster
    for idx = 1:n_c
        popu(best(idx,1),:) = centers_copy(idx,:);  
        fitness_popu(best(idx,1),1) = fit_values(idx,1);
    end       
    n_iteration = n_iteration +1;    
    % record the best fitness in each iteration
    best_fitness(n_iteration, 1) = min(fit_values);
    fprintf('RUN: %d \t subsetsize: %d \t Iter: %d \t Err: %.4f \t \n',run,dnf,n_iteration,best_fitness(n_iteration, 1))
end
   Best_fitness(1,dnf/dnd)= best_fitness(end, 1);
end
    Best_fit(run,:)=Best_fitness;
end
    BESTFIT=(1-mean(Best_fit))*100;
    save('MBSOFS','Best_fit','BESTFIT')
