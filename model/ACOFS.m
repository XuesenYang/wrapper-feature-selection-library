% Editor:Summer
% Institution: Shenzhen University
% E-mail:1348825332@qq.com
% date:2018-12-16
% 30 runs with 12 datasets after ACOFS
% 10-fold cross-validation with k-nn classifier
clc;
clear;
%% load classification datasets
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
     case 'ORL'
 load('ORL.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'COIL20'
 load('COIL20.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'orlraws10P'
 load('orlraws10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'pixraw10P'
 load('pixraw10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'warpAR10P'
 load('warpAR10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'warpPIE10P'
 load('warpPIE10P.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'Yale'
 load('Yale.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'GLIOMA'
 load('GLIOMA.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'lung_discrete'
 load('lung_discrete.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
     case 'colon'
 load('colon.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'lung'
 load('lung.mat')
    data=[X,Y];
    dn=5:5:50;
    dnsize=10;
    dnd=5;
    case 'ForestTypes'
 load('ForestTypes.mat')
    dn=2:2:20;
    dnsize=10;
    dnd=2;
end
%% Parameters setting
e=10; % e-fold cross-validation
group=data(:,end);
class=unique(data(:,end));
for i=1:length(class)
    sa=[];
    sa=data((group==class(i)),:);
    [number_of_smile_samples,~] = size(sa); % Column-observation
    smile_subsample_segments1 = round(linspace(1,number_of_smile_samples,e+1)); % indices of subsample segmentation points    
    data_group{i}=sa;
    smile_subsample_segments{i}=smile_subsample_segments1;
end
RHO=corr(data);
R=abs(RHO);
n=size(data,2)-1;   % The number of features without class
CrossOverProb=0.9;  % The Cross Over Probability
MutationProb=0.04;  % The Mutation Probability
GenerationNo = 100; % The Number of generation
na = 50;            % The Size of popuation
cc = 1;
maxrun=30;  %run times
   Best_fit=zeros(1,dnsize);
for run=1:maxrun
   Best_fitness=zeros(1,dnsize);
for dnf=dn
    m=dnf; % The number of selected features
        for i=1:n
            trail(i) = cc;
            deltatrail(i) = 0;
        end    
maxnoiteration = 100;
k = 50;
p = 1;
rho = 0.75;
alpha=1;
beta=1;
counter=1;   % The First iteration 
%%  Initialize population
for j=1:na
     Select = randperm(n);
     s(j,1:m)= Select(1:m);
end    
popu=s;
fv=zeros(1,m);
%% Divide training set and test set and calculate fitness value with 10-fold cross-validation
for jdx=1:na
   for i=1:e   
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; %training set
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];%test set
    end
    val=s(jdx,:);
    mdl = fitcknn(data_tr(:,val),data_tr(:,end),'NumNeighbors',4,'Standardize',1);%KNN-classifier
    Ac1=predict(mdl,data_ts(:,val)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
end
    E(1,jdx)=mean(Fit);
end
        
[MinError,MinErrorIndex]=min(E);
[SO,IX]=sort(E); %Output Individual Optimality

for j=1:k    
   for l=1:m
      for i=1:n
         deltatrail(i)=0;
         if (s(IX(j),l)==i)
            Errorg = max(SO(1:k))-E(IX(j));
            Errorhg=0;
            for h=1:k
              if ( (max(SO(1:k))-E(IX(h))) > Errorhg ) 
                Errorhg = max(SO(1:k))-E(IX(h));
                end
            end    
            if (Errorhg ~= 0)
              deltatrail(i)=Errorg/Errorhg; 
            else
              deltatrail(i)=1;
            end 
         end  
         trail(i)= rho * trail(i) + deltatrail(i);
      end 
   end 
end  
%% looping
while ( (counter < maxnoiteration) & (MinError > 0.001) )
s=popu;
counter = counter+1;
for j=1:na
    temp1(1:m)= s(IX(round(rand*(k-1))+1),1:m);
    r=randperm(m);
    for l=1:m-p
      temp2(l)=temp1(r(l));
    end        
    s(j,1:m-p)=temp2(1:m-p);
end    


for mm = m-p+1:m
  for j=1:na
   for i=1:n
     flag=0;
     for l=1:m-p
        if (s(j,l)==i)
          flag=1;
        end
     end  
     if (flag==1)
       USM(i)=0;
     else
       den=0;
       for l=1:m-p
         den=den+R(i,s(j,l));
       end  
       if (den~=0)
           LI(i)=R(i,n+1)/den;
       else
           LI(i)=R(i,n+1);
       end   
       vis(i)=LI(i)^beta;
       trail_p(i)=trail(i)^alpha;
       sigma=0;
       for ii=1:n
         flag=0;
         for l=1:m-p
           if(s(j,l)==ii)
             flag=1;
           end
         end  
         if (flag==0)
           den=0;
           for l=1:m-p
             den=den+R(ii,s(j,l));
           end  
           if (den~=0)
              LI(ii)=R(ii,n+1)/den;
           else
              LI(ii)=R(ii,n+1);
           end   
           sigma=sigma+ (trail(ii)^alpha) * (LI(ii)^beta);
         end  
       end  
       USM(i)=(vis(i)*trail_p(i))/sigma;  
       end  
   end  
   [maxf,maxfindex]=max(USM);
   s(j,mm)=maxfindex;
  end
end      


for j=1:na
    flag=0;
    for k1=1:m
        for k2=k1+1:m
            if(s(j,k1)==s(j,k2))
               flag=1;
            end
        end
    end  
    if (flag==1)
         Select=randperm(n);
         s(j,1:m)=Select(1:m);
    end  
end 
        


for i=1:na
  for j=i+1:na 
    if (s(i,1:m)==s(j,1:m))
      Select = randperm(n);
       s(j,1:m)= Select(1:m);
    end
  end
end 
    
for jdx=1:na
   for i=1:e    
    data_ts=[];data_tr =[];
    for j=1:length(class)
      smile_subsample_segments1=smile_subsample_segments{j};
      sa=data_group{j};
      test= sa(smile_subsample_segments1(i):smile_subsample_segments1(i+1) , :); % current_test_smiles
      data_ts=[test;data_ts] ; 
      train = sa;
      train(smile_subsample_segments1(i):smile_subsample_segments1(i+1),:) = [];
      data_tr =[train;data_tr];
    end
    val=s(jdx,:);
    mdl = fitcknn(data_tr(:,val),data_tr(:,end),'NumNeighbors',4,'Standardize',1);
    Ac1=predict(mdl,data_ts(:,val)); 
    Fit(i)=sum(Ac1~=data_ts(:,end))/size(data_ts,1);
   end
       fv(1,jdx)=mean(Fit);
    if fv(1,jdx)<E(1,jdx)  %Update optimal
        E(1,jdx)=fv(1,jdx);
        popu(jdx,:)=val;
    end
end
        
[MinError,MinErrorIndex]=min(E);
[SO,IX]=sort(E);

fprintf('RUN: %d \t subsetsize: %d \t Iter: %d \t Err: %.4f \t \n',run,dnf,counter,MinError)
for j=1:k
   for l=1:m
      for i=1:n
         deltatrail(i)=0;
         if (s(IX(j),l)==i)
            Errorg = max(SO(1:k))-E(IX(j));
            Errorhg=0;
            for h=1:k
              if ( (max(SO(1:k))-E(IX(h))) > Errorhg ) 
                Errorhg = max(SO(1:k))-E(IX(h));
                end
            end    
            if (Errorhg ~= 0)
              deltatrail(i)=Errorg/Errorhg; 
            else
              deltatrail(i)=1;
            end 
         end  
         trail(i)= rho * trail(i) + deltatrail(i);
      end 
   end
end 


%for j=1:na
%    temp1(1:m)= s(IX(round(rand*(k-1))+1),1:m);
%    r=randperm(m);
%    for l=1:m-p
%     temp2(l)=temp1(r(l));
%    end;        
%    s(j,1:m-p)=temp2(1:m-p);
%end;    

end    
Best_fitness(1,dnf/dnd)= MinError;
end
Best_fit(run,:)=Best_fitness;
end
%% Store the final results
    BESTFIT=(1-mean(Best_fit))*100;
    save('ACOFS','Best_fit','BESTFIT') 

