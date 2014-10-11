%% Learning
%Aquest script compara les diferents tècniques de Machine Learning
%Autors :   Noè Rosanas Boeta noe.rosanas@gmail.com
%           Claudi Ruiz Camps claudi_ruiz@hotmail.com

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NAIVE BAYES,LDA, TREE I NEURAL NETWORK (MITJANï¿½ANT PCA)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% definim targets coneguts de les mesures d'entrenament
t=zeros(numeros,mostres*numeros);
t1=zeros(1,mostres*numeros);
close all
for i=1:mostres*numeros 
    
    if 1<=i && i<=1*mostres
       t(:,i)=[0;0;0;0;0;0;0;0;0;1]; %0
       t1(1,i)=0; %0
    
    elseif 1*mostres<i && i<=2*mostres
       t(:,i)=[0;0;0;0;0;0;0;0;1;0]; %1
       t1(1,i)=1; %1
    
    elseif 2*mostres<i && i<=3*mostres
       t(:,i)=[0;0;0;0;0;0;0;1;0;0]; %2
       t1(1,i)=2; %2
       
    elseif 3*mostres<i && i<=4*mostres
       t(:,i)=[0;0;0;0;0;0;1;0;0;0]; %3
       t1(1,i)=3; %3       
       
    elseif 4*mostres<i && i<=5*mostres
       t(:,i)=[0;0;0;0;0;1;0;0;0;0]; %4
       t1(1,i)=4; %4
       
    elseif 5*mostres<i && i<=6*mostres
       t(:,i)=[0;0;0;0;1;0;0;0;0;0]; %5
       t1(1,i)=5; %5
       
    elseif 6*mostres<i && i<=7*mostres
       t(:,i)=[0;0;0;1;0;0;0;0;0;0]; %6
       t1(1,i)=6; %6
       
    elseif 7*mostres<i && i<=8*mostres
       t(:,i)=[0;0;1;0;0;0;0;0;0;0]; %7
       t1(1,i)=7; %7
       
    elseif 8*mostres<i && i<=9*mostres
       t(:,i)=[0;1;0;0;0;0;0;0;0;0]; %8
       t1(1,i)=8; %8
       
    elseif 9*mostres<i && i<=10*mostres
       t(:,i)=[1;0;0;0;0;0;0;0;0;0]; %9
       t1(1,i)=9; %9
    end       
end
%% Assignem una certa quantitat del total de les 100 mostres originals per a entrenar i testejar (validar no ja que trainbr no fa validacio)
[trainInd,valInd,testInd] = dividerand(mostres*numeros,0.8,0.,0.2);
%% Seleccionem les mostres simulades d'entrenament amb els seus targets corresponents. 
if mult~=0 
    simaux=[];
    tsimaux=[];
    t1simaux=[];
    for i=1:mult
       simaux=[siminput(:,trainInd) simaux]; 
       tsimaux=[t(:,trainInd) tsimaux]; 
       t1simaux=[t1(:,trainInd) t1simaux]; 
       size(simaux)
    end
    clear siminput
    siminput=simaux;
    tsim=tsimaux;
    t1sim=t1simaux;
end
if mult==0
    siminput=[];
    tsim=[];
    t1sim=[];
end
ntrain=size([input(:,trainInd) siminput],2);
ntest=size(input(:,testInd),2);
%% PCA, util per a obtenir una quantitat mes reduida de features pero alhora mes utils, per tal de facilitar la feina als classificadors
% PCA_projinput: Projection of the data A into the PCA space
[evec,PCA_projinput,eval] = princomp([input(:,trainInd) siminput]');
% Check: Projection of data in the PCA basis is given by (A-mean(A))*evec:
projtrain =  ([input(:,trainInd) siminput]'-ones(ntrain,1)*mean([input(:,trainInd) siminput]'))*evec;
projtest  =  (input(:,testInd)'-ones(ntest,1)*mean(input(:,testInd)'))*evec;
inputtrain=projtrain(:,1:pca)'; %Inputs d'entrenament post PCA
inputtest=projtest(:,1:pca)'; % Inputs de testeig post PCA
figure;
plot(cumsum(100*eval/sum(eval)),'ro') % Quantitat d'informaciï¿½ que ens proporcionen les dimensions del PCA, mes informacio no implica millors resultats
axis([0 max(temps1) min(x) max(x)]);
xlabel('PCA dimensions')
ylabel('Information percentage')
axis([0 15 0 100]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CLASSIFICADORS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Preparem algunes variables pels classificadors
intervals=zeros(2,features);
N = size(inputtest',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NAIVE BAYES GAUSSIAN DISTRIBUTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
display('Naive Bayes GD')
nbGau= NaiveBayes.fit(inputtrain',[t1(:,trainInd)';t1sim']);
%% Calcul d'error d'entrenament
nbGauClass= nbGau.predict(inputtrain');
bad = sum(nbGauClass~=[t1(:,trainInd)';t1sim']);
nbGauResubErr = 100*bad / ntrain
[nbGDResubCM,Order] = confusionmat([t1(:,trainInd)';t1sim'],nbGauClass);
nbGDResubCM;
%% Calcul d'error de testeig
nbGauClass= nbGau.predict(inputtest');
bad = sum(nbGauClass~=t1(:,testInd)');
nbGauResubErr = 100*bad / ntest
[nbGDResubCM,Order] = confusionmat(t1(:,testInd)',nbGauClass);
nbGDResubCM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NAIVE BAYES KERNEL DENSITY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
display('Naive Bayes KD')
nbKD= NaiveBayes.fit(inputtrain',[t1(:,trainInd)';t1sim'],'dist','kernel');
%% Calcul d'error d'entrenament
nbKDClass= nbKD.predict(inputtrain');
bad = sum(nbKDClass~=[t1(:,trainInd)';t1sim']);
nbKDResubErr = 100*bad / ntrain
[nbKDResubCM,Order] = confusionmat([t1(:,trainInd)';t1sim'],nbKDClass);
nbKDResubCM;
%% Calcul d'error de testeig
nbKDClass= nbKD.predict(inputtest');
bad = sum(nbKDClass~=t1(:,testInd)');
nbKDResubErr = 100*bad / ntest
[nbKDResubCM,Order] = confusionmat(t1(:,testInd)',nbKDClass);
nbKDResubCM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LDA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
display('LDA')
%% Calcul d'error d'entrenament
ldaClass = classify(inputtrain',inputtrain',[t1(:,trainInd)';t1sim'],'linear');
bad = sum(ldaClass~=[t1(:,trainInd)';t1sim']);
ldaResubErr = 100*bad / ntrain
[ldaResubCM,Order] = confusionmat([t1(:,trainInd)';t1sim'],ldaClass);
ldaResubCM;
%% Calcul d'error de testeig
ldaClass = classify(inputtest',inputtrain',[t1(:,trainInd)';t1sim'],'linear');
bad = sum(ldaClass~=t1(:,testInd)');
ldaResubErr = 100*bad / ntest
[ldaResubCM,Order] = confusionmat(t1(:,testInd)',ldaClass);
ldaResubCM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TREE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
tree = classregtree(inputtrain',[t1(:,trainInd)';t1sim']);
display('TREE')
%% Calcul d'error d'entrenament
dtClass = tree.eval(inputtrain');
bad = sum(dtClass~=[t1(:,trainInd)';t1sim']);
dtResubErr = 100*bad / ntrain
[dtResubCM,Order] = confusionmat([t1(:,trainInd)';t1sim'],dtClass);
dtResubCM;
%% Calcul d'error de testeig
dtClass = tree.eval(inputtest');
bad = sum(dtClass~=t1(:,testInd)');
dtResubErr = 100*bad / ntest
[dtResubCM,Order] = confusionmat(t1(:,testInd)',dtClass);
dtResubCM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NEURAL NETWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
display('NEURAL NETWORK')
net = patternnet(conos);
[net,tr] = trainbr(net,inputtrain,[t(:,trainInd) tsim]);
nntraintool
figure
plotconfusion([t(:,trainInd) tsim],net(inputtrain));
title('Train')
y = net(inputtest);
%  print -depsc -tiff -r300 confusiontrain
figure
plotconfusion(t(:,testInd),y)
title('Test')
tests = t(:,testInd);
% print -depsc -tiff -r300 confusiontest
% plotroc(net)

