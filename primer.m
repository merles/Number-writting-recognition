%% Data Extraction
%Aquest fitxer és l'encarregat d'extreure les dades del fitxers
%enregistrats pel mòbil 
%Autors :   Noè Rosanas Boeta noe.rosanas@gmail.com
%           Claudi Ruiz Camps claudi_ruiz@hotmail.com

clear all;
close all;
clc;
conos=15;
pca=7;
mostres=10;     % quantitat de mostres per cada numero
numeros=10;     % quantitat de numeros (0,1,2,etc)
k=1;            % contador de columnes per la variable input
mult=1;         % quantitat de versions simulades de les mostres d'entrenament (multiple de quantitat de mostres d'entrenament)
                % util si no es tenen les suficients mesures
features=24;    % quantitat de caracteristiques
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXTRACCIO DE DADES I CREACIO DE LES FEATURES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
for numero=1:numeros   
    for mostra=1:mostres
        fid =fopen([num2str(numero-1),num2str(mostra),'.txt'],'r');
        dades_fitxer=textscan(fid,'%f %f %f %f','headerlines',1);
%         display([num2str(numero-1),num2str(mostra),'.txt'])
        dades_fitxer=cell2mat(dades_fitxer);
        
        
        mida=size(dades_fitxer,1);
        fclose(fid); 
        x=dades_fitxer(:,1);
        y=dades_fitxer(:,2);
        z=dades_fitxer(:,3);
        temps=dades_fitxer(:,4);
        temps1(1)=temps(1);
        for i=1:size(temps,1)-1
            temps1(i+1)=temps1(i)+temps(i+1);
        end   
        if mostra==1 && numero==1
            figure
            plot(temps1,x)
            axis([0 max(temps1) min(x) max(x)]);
            xlabel('Time')
            ylabel('Magnitude (g)')
        end
        
        countx=1;county=1;countz=1;
        input(:,k)=[...
                        mean(x);...
                        mean(y);...
                        mean(z);...
                        std(x);...
                        std(y);...
                        std(z);...
                        var(x);...
                        var(y);...
                        var(z);...
                        iqr(x);...
                        iqr(y);...
                        iqr(z);...
                        mad(x);...
                        mad(y);...
                        mad(z);... 
                        corr(x,y);...
                        corr(x,z);...
                        corr(z,y);...
                        max(x);...
                        max(y);...
                        max(z);...
                        min(x);...
                        min(y);...
                        min(z);...
                        ];
                    k=k+1;
    end
end
%% Simulacio de mostres d'entrenament
for k=1:mult
    for numero=1:numeros
        for mostra=1:mostres
            for feature=1:features
                auxiliar=input(feature,(numero-1)*mostres+1:numero*mostres);
                siminput(feature,mostra+ mostres*(numero-1)+mostres*numeros*(k-1))=...
                    (mean(auxiliar)-std(auxiliar)+2*rand(1)*std(auxiliar));
            end
        end
    end
end
segon;
