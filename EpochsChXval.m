Matlab clear;
clc;

%% Initial Values...

%%----Dataset Folder Address
DatasetFold = ['\\VSD_EEG_motion\Data\Participant_EEG_data\P12 Analysis\Segments\'];

%%----ERP features
BaselineDuration = 0.2; % Baseline correction (in seconds)

ERPs(1).Tag = 'P100';
ERPs(1).Ranges = [0.070 0.205]+BaselineDuration;
ERPs(1).NominalTime = [0.100 + BaselineDuration];
ERPs(1).Regions = {'Frontal';'Central';'Parietal-Occipital'};
ERPs(1).RegionsChann = {[];[22,23];[14:20]};

ERPs(2).Tag = 'P200-P300';
ERPs(2).Ranges = [0.220 0.700]+BaselineDuration;
ERPs(2).NominalTime = [0.300 + BaselineDuration];
ERPs(2).Regions = {'Frontal';'Central';'Parietal-Occipital'};
ERPs(2).RegionsChann = {[];[6:9,28,29,22:26,11,12];[11:20,22,23,26]};

ERPs(3).Tag = 'N400';
ERPs(3).Ranges = [0.400 0.900]+BaselineDuration;
ERPs(3).NominalTime = [0.400 + BaselineDuration];
ERPs(3).Regions = {'Frontal';'Central';'Parietal-Occipital'};
ERPs(3).RegionsChann = {[1:4,32];[6,7,11,22,28,29];[13:20]};

ERPs(4).Tag = 'N200';
ERPs(4).Ranges = [0.120 0.300]+BaselineDuration;
ERPs(4).NominalTime = [0.200 + BaselineDuration];
ERPs(4).Regions = {'Frontal';'Central';'Parietal-Occipital'};
ERPs(4).RegionsChann = {[1:4];[6:9,11,12,22:26,28,29];[13:20]};
%%        
Paradigms = {};
Paradigms(1).Tag = 'A';
Paradigms(2).Tag = 'S';

%%----General Regions (Electrode clusters)
% Regions.Chann = {[1:9,11:17,24];[13,17:20,22:32];[1:7,27:32];[7,8,12,23:25,29];[6,8,9,11,26,28,25,22];[12:15,19,20,23];[13,16:18];[1:9,11:20,22:32]};
% Regions.Labels = {'Right';'Left';'Frontal';'Central';'Temporal';'Parietal';'Occipital';'General'};
% 
Regions.Chann = {[1:4,32];[6:9,11,12,22:26,28,29];[11:20,22,23,26]};
Regions.Labels = {'Frontal';'Central';'Parietal-Occipital'};
        
% 1) a frontal bin (average for the F elects), 
% 2) centro-parietal (average of the FC/C/CP/P elecs) and 
% 3) Occipital bin (average o elec).

%% Outputs 
NumHighRankedFeatures = 128/2;
NumIterations = 5;
FSType = 0;     % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
                %         otherwise (e.g 0)=all features

RawEpochsChannels = [Regions.Chann{:}];
% RawEpochsChannels = [1:32];
NumHighRnkFeat = [128 64 32 16];
KFoldsss = [2, 5, 10];
for HRFeats = 1:1
for FSType = 0:0
    for Kfo = 3 %Only generates k=10 output
           Kfolds = KFoldsss(Kfo);
           NumHighRankedFeatures = NumHighRnkFeat(HRFeats);
XLOutputName = ['Results_RawEpochs_K',num2str(Kfolds),'_FS',num2str(FSType),'_NumRnkFS',num2str(NumHighRankedFeatures),'_',datestr(now,'mmmm-dd-yyyy_HH_MM_SS'),'.xlsx'];

%% Excel of ERPs per sheet for all electrodes

XLLine = 1;
xlswrite(XLOutputName,{'PT','Paradigm','Avg ACCs','TPR','TNR','AUC','StdDev ACCs','Min ACCs','Max ACCs','ACCs:'},'SWLDA-UnBal',['A' num2str(XLLine)]); %Names sheet of output file 'SWLDA-UnBal' and adds labels in first row

for pt = 1:12
    for o = 1:length(Paradigms)

            tmpfileAdd = [DatasetFold 'P' num2str(pt) '_gridmotion_Artifact_Rejection_T' Paradigms(o).Tag '.mat'];
            [PtSegments{pt,1}, ChLabels, Fs, ~] = ImportingBCIData(tmpfileAdd);

            tmpfileAdd = [DatasetFold 'P' num2str(pt) '_gridmotion_Artifact_Rejection_NT' Paradigms(o).Tag '.mat'];
            [PtSegments{pt,2}, ChLabels, Fs, ChannInfo] = ImportingBCIData(tmpfileAdd);

            Paradigms(o).PeakFeatures = [];
            Paradigms(o).LatencyFeatures = [];
            Paradigms(o).AvgsFeatures = [];
            Paradigms(o).RawEpochFeatures = zeros(size(PtSegments{pt,1},1)+size(PtSegments{pt,2},1),length(RawEpochsChannels)*size(PtSegments{pt,1}(:,:,BaselineDuration*Fs+1:end),3));
%             Paradigms(o).RawEpochFeatures = [reshape(PtSegments{pt,1}(:,RawEpochsChannels,BaselineDuration*Fs+1:end),[size(PtSegments{pt,1},1),length(RawEpochsChannels)*size(PtSegments{pt,1}(:,:,BaselineDuration*Fs+1:end),3)]);...
%                                              reshape(PtSegments{pt,2}(:,RawEpochsChannels,BaselineDuration*Fs+1:end),[size(PtSegments{pt,2},1),length(RawEpochsChannels)*size(PtSegments{pt,2}(:,:,BaselineDuration*Fs+1:end),3)])];
            for tmp1 = 1:size(PtSegments{pt,1},1)
                tmpseg = [];
                for tmp2 = 1:length(RawEpochsChannels)
                    tmpseg = [tmpseg squeeze(PtSegments{pt,1}(tmp1,tmp2,BaselineDuration*Fs+1:end))'];
                end
                Paradigms(o).RawEpochFeatures(tmp1,:) = tmpseg;
            end
            for tmp1 = 1:size(PtSegments{pt,2},1)
                tmpseg = [];
                for tmp2 = 1:length(RawEpochsChannels)
                    tmpseg = [tmpseg squeeze(PtSegments{pt,2}(tmp1,tmp2,BaselineDuration*Fs+1:end))'];
                end
                    Paradigms(o).RawEpochFeatures(tmp1+size(PtSegments{pt,1},1),:) = tmpseg;
            end
                for E = 1:length(ERPs)
                    Paradigms(o).TargAvgs{E} = mean(PtSegments{pt,1}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),3);
                    Paradigms(o).NonTargAvgs{E} = mean(PtSegments{pt,2}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),3);
                    if ERPs(E).Tag(1) =='P'
                        [Paradigms(o).TargPeaks{E},Ind] = max(PtSegments{pt,1}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),[],3);
                        Paradigms(o).TargLatency{E} = ((Ind - 1)/ Fs) + ERPs(E).Ranges(1) - BaselineDuration;
                        [Paradigms(o).NonTargPeaks{E},Ind] = max(PtSegments{pt,2}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),[],3);
                        Paradigms(o).NonTargLatency{E} = ((Ind - 1)/ Fs) + ERPs(E).Ranges(1) - BaselineDuration;
                    else
                        [Paradigms(o).TargPeaks{E},Ind] = min(PtSegments{pt,1}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),[],3);
                        Paradigms(o).TargLatency{E} = ((Ind - 1)/ Fs) + ERPs(E).Ranges(1) - BaselineDuration;
                        [Paradigms(o).NonTargPeaks{E},Ind] = min(PtSegments{pt,2}(:,:,ceil(ERPs(E).Ranges(1)*Fs):floor(ERPs(E).Ranges(2)*Fs)),[],3);
                        Paradigms(o).NonTargLatency{E} = ((Ind - 1)/ Fs) + ERPs(E).Ranges(1) - BaselineDuration;
                    end
                    Paradigms(o).AvgsFeatures = [Paradigms(o).AvgsFeatures , [Paradigms(o).TargAvgs{E};Paradigms(o).NonTargAvgs{E}]];
                    Paradigms(o).PeakFeatures = [Paradigms(o).PeakFeatures , [Paradigms(o).TargPeaks{E};Paradigms(o).NonTargPeaks{E}]];
                    Paradigms(o).LatencyFeatures = [Paradigms(o).LatencyFeatures , [Paradigms(o).TargLatency{E};Paradigms(o).NonTargLatency{E}]];
            end

            Paradigms(o).SegmentsLabels = [ones(size(Paradigms(o).TargPeaks{E},1),1) ; zeros(size(Paradigms(o).NonTargPeaks{E},1),1)];

            XLLine = XLLine+1; %Increments to next line for outputting into the Excel spreadsheet
        
%% Selected features based on the visual investigation
 
        Paradigms(o).SelectedFeatures = [];
        Paradigms(o).SelectedAvgsFeatures = [];
%         Paradigms(o).SelectedFeatures = [];
        for E = 1:length(ERPs)
            tmpSelectChann = [ERPs(E).RegionsChann{:}];
            Paradigms(o).SelectedFeatures = [Paradigms(o).SelectedFeatures, [Paradigms(o).TargPeaks{E}(:,tmpSelectChann);Paradigms(o).NonTargPeaks{E}(:,tmpSelectChann)]];     
            Paradigms(o).SelectedAvgsFeatures = [Paradigms(o).SelectedAvgsFeatures, [Paradigms(o).TargAvgs{E}(:,tmpSelectChann);Paradigms(o).NonTargAvgs{E}(:,tmpSelectChann)]];     
        end

%% --------------------------------------------------------------------------------
%%------------ Epochs(whole segments) ---------------------------------------------
%%---------------------------(Use the raw EEG segments for classifications)--------
%%---------------------------------------------------------------------------------
          %% Unbalanced Dataset
        clc;
        BalancedDataset = 1;

        fprintf('Balanced Datasets : \n');
        
        %%--- SWLDA
        Steps = 60; PEnter = 0.1; Display = 'off';
        ACCs = [];
        Kappas = [];
        TPRs = []; TNRs = [];
        Scores = [];
        TrueLabels = [];

        for k = 1:NumIterations
            [Accuracies, Kappa, CMs, TPR, TNR, Score, TrueLabel] = SWLDAKfolds (Paradigms(o).RawEpochFeatures,Paradigms(o).SegmentsLabels,Kfolds,Steps,PEnter,Display,BalancedDataset);
            ACCs = [ACCs Accuracies];  Kappas = [Kappas Kappa]; TPRs = [TPRs TPR]; TNRs = [TNRs TNR]; Scores = [Scores ; Score]; TrueLabels = [TrueLabels ; TrueLabel];
        end
%         RepTestLabels = repmat(TrueLabels,k,1);
        [~,~,~,AUCsvm] = perfcurve(TrueLabels,Scores(:,2),1);

        fprintf('Average SWLDA crossvalidation accuracies = %.2f \n',mean(ACCs)*100);
        ACCs_str = split(cellstr(num2str(ACCs)))';
        tmpRes = {['Pt_' num2str(pt)] , Paradigms(o).Tag ,...
                  num2str(mean(ACCs)*100) , num2str(mean(TPRs)*100), num2str(mean(TNRs)*100),num2str(AUCsvm),...
                  num2str(std(ACCs)*100), num2str(min(ACCs(:))*100), num2str(max(ACCs(:))*100)};
        xlswrite(XLOutputName,[tmpRes,ACCs_str] , 'SWLDA-Bal', ['A' num2str(XLLine)]);
        
        %% Balanced
        BalancedDataset = 1;

        fprintf('Balanced Datasets : \n');
%%  
    end
        
end
end
    end
end

%% Done        
disp('---------DONE--------');
%%

%% Functions
function TNR = CM2TNR(CM)
%     TNR = TN / (TN+FP)  %Specificity
    TNR = CM(1,1) / (CM(1,1) + CM(1,2));
%      TP = CM(2,2), TN = CM(1,1), FP = ,CM(1,2), FN = CM(2,1)
end

function TPR = CM2TPR(CM)
%     TPR = TP / (TP+FN)  %Sensitivity
    TPR = CM(2,2) / (CM(2,2) + CM(2,1));
%      TP = CM(2,2), TN = CM(1,1), FP = ,CM(1,2), FN = CM(2,1)
end

function Kappa = CM2Kappa(CM)
%     Kappa = (2*(TP*TN - FN*FP))/((TP+FP)*(FP+TN) + (TP+FN)*(fn+tn))));
    Kappa = (2*(CM(2,2)*CM(1,1) - CM(2,1)*CM(1,2)))/((CM(2,2)+CM(1,2))*(CM(1,2)+CM(1,1)) + (CM(2,2)+CM(2,1))*(CM(2,1)+CM(1,1)));
%      TP = CM(2,2), TN = CM(1,1), FP = ,CM(1,2), FN = CM(2,1)
end

%Classification with Feature selection
function [Accuracies, Kappas, CMs] = KfoldQDAFS (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
    
    
    Accuracies = [];
    Kappas = [];
    CMs = [];
    
    %QDA
    disp([num2str(Kfolds) '-Fold QDA crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        switch FSType
            case 1
                [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
            case 2
                [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
            case 3
                [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
            otherwise
                idx = [1:size(Dataset,2)];
                NumHighRankedFeatures = size(Dataset,2);
        end
        
        MdlLinear = fitcdiscr(Dataset(train,idx(1:NumHighRankedFeatures)),Labels(train,:),'DiscrimType','pseudoQuadratic'); 
%         'diagquadratic' 'pseudoquadratic'

        Labels_Predict = predict(MdlLinear,Dataset(test,idx(1:NumHighRankedFeatures)));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        CMs = [CMs ; CM];
    end
end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldLDAFS (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
    
    Accuracies = [];
    Kappas = [];
    TNRs = [];
    TPRs = [];
    CMs = [];
    
    %LDA
    disp([num2str(Kfolds) '-Fold LDA crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        switch FSType
            case 1
                [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
            case 2
                [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
            case 3
                [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
            otherwise
                idx = [1:size(Dataset,2)];
                NumHighRankedFeatures = size(Dataset,2);
        end
        
        MdlLinear = fitcdiscr(Dataset(train,idx(1:NumHighRankedFeatures)),Labels(train,:));
%         'linear' 'diaglinear' 'pseudolinear'

        Labels_Predict = predict(MdlLinear,Dataset(test,idx(1:NumHighRankedFeatures)));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end
end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldLDAFS5FoldTraining (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
    
    Accuracies = [];
    Kappas = [];
    TNRs = [];
    TPRs = [];
    CMs = [];
    
    %LDA
    disp([num2str(Kfolds) '-Fold LDA crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        switch FSType
            case 1
                [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
            case 2
                [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
            case 3
                [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
            otherwise
                idx = [1:size(Dataset,2)];
                NumHighRankedFeatures = size(Dataset,2);
        end
        tmpTrainACC = [];
        for ft = 1:length(idx)
            MdlLinear = fitcdiscr(Dataset(train,idx(1:ft)),Labels(train,:));
            Labels_Predict = predict(MdlLinear,Dataset(train,idx(1:ft)));
            acc = 1 - (sum(abs(Labels(train,:)-Labels_Predict))/length(Labels(train,:)));
            tmpTrainACC(ft) = acc;
        end
%         [~,ii] = max(tmpTrainACC);
%         NumHighRankedFeatures = ii;
        DifftmpTrainACC = diff(tmpTrainACC);
        idx(find(DifftmpTrainACC<=0)+1) = [];
%         MdlLinear = fitclinear(Dataset(train,idx(1:NumHighRankedFeatures)),Labels(train,:));
        MdlLinear = fitcdiscr(Dataset(train,idx),Labels(train,:));
        
        Labels_Predict = predict(MdlLinear,Dataset(test,idx));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end
end

function [Accuracies, Kappas, CMs, TPRs, TNRs, Predicted_Scores, TrueLabels] = SWLDAKfolds (Dataset,Labels,Kfolds,Steps,PEnter,Display,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
    
    Accuracies = [];
    Kappas = [];
    TNRs = [];
    TPRs = [];
    CMs = [];
    Predicted_Scores = [];
    TrueLabels = [];
    
    %SWLDA
    disp(['SWLDA(linear regression) classification: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        if Steps == 0
            [B,SE,PVAL,in,stats] = stepwisefit(Dataset(train,:),Labels(train,:),'Display',Display,'PEnter',PEnter);
        else
            [B,SE,PVAL,in,stats] = stepwisefit(Dataset(train,:),Labels(train,:),'MaxIter', Steps,'Display',Display,'PEnter',PEnter);
        end

        
%         idx1= find(Labels(train)==1);
%         idx2= find(Labels(train)==0);
%         Bias = -B(in)'*(mean(Dataset(idx1,in))+mean(Dataset(idx2,in)))'/2;
        
      
%         ew = Dataset(test,in)*B(in) + Bias;
%         
%         Labels_Predict = ceil(ew);
%         Predicted_Score = zeros(length(Labels_Predict),2);
%         Predicted_Score = [abs((ew-1)/2),(ew+1)/2];
        
        MdlLinear = fitclinear(Dataset(train,in),Labels(train,:));

        [Labels_Predict, Predicted_Score] = predict(MdlLinear,Dataset(test,in));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
        Predicted_Scores = [Predicted_Scores ; Predicted_Score];
        TrueLabels = [TrueLabels ; Labels(test,:)];
    end
    
end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldLDAFS5FoldOptTraining (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced,TrainingKfolds)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
      
    Accuracies = [];
    Kappas = [];
    TPRs = [];
    TNRs = [];
    CMs = [];
    % SVM
    disp([num2str(Kfolds) '-Fold LDA crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
%         switch FSType
%             case 1
%                 [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
%             case 2
%                 [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
%             case 3
%                 [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
%             otherwise
%                 idx = [1:size(Dataset,2)];
%                 NumHighRankedFeatures = size(Dataset,2);
%         end
        
        tmpTrainDataset = Dataset(train,:); tmpTrainLabels = Labels(train,:); 
        tmpindices = crossvalind('Kfold',tmpTrainLabels,TrainingKfolds);
        MDs = {};
        tmpTrainACC = [];
        for tmpi = 1:TrainingKfolds
            tmptestidx = (tmpindices == tmpi);
            tmptrainidx = ~tmptestidx;
            
            MDs{tmpi} = fitcdiscr(tmpTrainDataset(tmptrainidx,:),tmpTrainLabels(tmptrainidx,:));
            Labels_Predict = predict(MDs{tmpi},tmpTrainDataset(tmptestidx,:));
            acc = 1 - (sum(abs(tmpTrainLabels(tmptestidx,:)-Labels_Predict))/length(tmpTrainLabels(tmptestidx,:)));
            tmpTrainACC(tmpi) = acc;
        end
        
        [~,OptIdx] = max(tmpTrainACC);
        
        Labels_Predict = predict(MDs{OptIdx},Dataset(test,:));
        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end

end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldSVMFS (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
      
    Accuracies = [];
    Kappas = [];
    TNRs = [];
    TPRs = [];
    CMs = [];
    % SVM
    disp([num2str(Kfolds) '-Fold SVM crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        switch FSType
            case 1
                [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
            case 2
                [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
            case 3
                [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
            otherwise
                idx = [1:size(Dataset,2)];
                NumHighRankedFeatures = size(Dataset,2);
        end
        
        MdlLinear = fitclinear(Dataset(train,idx(1:NumHighRankedFeatures)),Labels(train,:));

        Labels_Predict = predict(MdlLinear,Dataset(test,idx(1:NumHighRankedFeatures)));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end

end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldSVMFS5FoldTraining (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
      
    Accuracies = [];
    Kappas = [];
    TPRs = [];
    TNRs = [];
    CMs = [];
    % SVM
    disp([num2str(Kfolds) '-Fold SVM crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        switch FSType
            case 1
                [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
            case 2
                [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
            case 3
                [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
            otherwise
                idx = [1:size(Dataset,2)];
                NumHighRankedFeatures = size(Dataset,2);
        end
        
        tmpTrainACC = [];
        for ft = 1:length(idx)
            MdlLinear = fitclinear(Dataset(train,idx(1:ft)),Labels(train,:));
            Labels_Predict = predict(MdlLinear,Dataset(train,idx(1:ft)));
            acc = 1 - (sum(abs(Labels(train,:)-Labels_Predict))/length(Labels(train,:)));
            tmpTrainACC(ft) = acc;
        end
%         [~,ii] = max(tmpTrainACC);
%         NumHighRankedFeatures = ii;
        DifftmpTrainACC = diff(tmpTrainACC);
        idx(find(DifftmpTrainACC<=0)+1) = [];
%         MdlLinear = fitclinear(Dataset(train,idx(1:NumHighRankedFeatures)),Labels(train,:));
        MdlLinear = fitclinear(Dataset(train,idx),Labels(train,:));
        
        Labels_Predict = predict(MdlLinear,Dataset(test,idx));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end

end

function [Accuracies, Kappas, CMs, TPRs, TNRs] = KfoldSVMFS5FoldOptTraining (Dataset,Labels,Kfolds,NumHighRankedFeatures,FSType,Balanced,TrainingKfolds)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
      
    Accuracies = [];
    Kappas = [];
    TPRs = [];
    TNRs = [];
    CMs = [];
    % SVM
    disp([num2str(Kfolds) '-Fold SVM crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
%         switch FSType
%             case 1
%                 [idx,scores] = fscmrmr(Dataset(train,:),Labels(train,:));
%             case 2
%                 [idx,scores] = fscchi2(Dataset(train,:),Labels(train,:));
%             case 3
%                 [idx,scores] = fsrftest(Dataset(train,:),Labels(train,:));
%             otherwise
%                 idx = [1:size(Dataset,2)];
%                 NumHighRankedFeatures = size(Dataset,2);
%         end
        
        tmpTrainDataset = Dataset(train,:); tmpTrainLabels = Labels(train,:); 
        tmpindices = crossvalind('Kfold',tmpTrainLabels,TrainingKfolds);
        MDs = {};
        tmpTrainACC = [];
        for tmpi = 1:TrainingKfolds
            tmptestidx = (tmpindices == tmpi);
            tmptrainidx = ~tmptestidx;
            
            MDs{tmpi} = fitclinear(tmpTrainDataset(tmptrainidx,:),tmpTrainLabels(tmptrainidx,:));
            Labels_Predict = predict(MDs{tmpi},tmpTrainDataset(tmptestidx,:));
            acc = 1 - (sum(abs(tmpTrainLabels(tmptestidx,:)-Labels_Predict))/length(tmpTrainLabels(tmptestidx,:)));
            tmpTrainACC(tmpi) = acc;
        end
        
        [~,OptIdx] = max(tmpTrainACC);
        
        Labels_Predict = predict(MDs{OptIdx},Dataset(test,:));
        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        TNRs = [TNRs CM2TNR(CM)];
        TPRs = [TPRs CM2TPR(CM)];
        CMs = [CMs ; CM];
    end

end

function [Accuracies, Kappas, CMs] = KfoldANN (Dataset,Labels,Kfolds,Balanced,ANNStruct)

    % Balanced: 1=Balanced dataset, 2=Unbalanced
  
    % FSType: 1=fscmrmr 2=fscchi2 3=fsrftest ; 
    %         otherwise (e.g 0)=all features
    
    
    if Balanced == 1
        Labelstmp = Labels; Datasettmp = Dataset; clear Labels Dataset
        
        tmpidx = randsample(find(Labelstmp==0),length(find(Labelstmp==1)));
        tmpidx = [find(Labelstmp==1) ; tmpidx];

        Labels = Labelstmp(tmpidx,:);
        Dataset = Datasettmp(tmpidx,:);
    end
      
    Accuracies = [];
    Kappas = [];
    CMs = [];
    % SVM
    disp([num2str(Kfolds) '-Fold ANN crossvalidation: ']);
    indices = crossvalind('Kfold',Labels,Kfolds);

    for i = 1:Kfolds
        test = (indices == i); 
        train = ~test;
        
        MdlANN = fitcnet(Dataset(train,:),Labels(train,:),"LayerSizes",ANNStruct);
        
        Labels_Predict = predict(MdlANN,Dataset(test,:));

        acc = 1 - (sum(abs(Labels(test,:)-Labels_Predict))/length(Labels(test,:)));
        [CM,order] = confusionmat(Labels(test,:),Labels_Predict,'Order',[0 1]);
%         cm = confusionchart(Labels(test,:),Labels_Predict);

        fprintf('Acc = %.3f, TP = %i, TN = %i, FP = %i, FN = %i.\n',acc,CM(2,2),CM(1,1),CM(1,2),CM(2,1));
        Accuracies = [Accuracies acc];
        Kappas = [Kappas CM2Kappa(CM)];
        CMs = [CMs ; CM];
    end

end

%Changing Structure of BCI Average signals structure to Segments:
function [Segments, ChLabels, Fs, ChannInfo] = ImportingBCIAvgDatatoSegments(Filename)
    load(Filename);

    Fs = SampleRate;
    Segments = [];
    ChLabels = cell(ChannelCount,1);
    for ch = 1:ChannelCount
        Segments(:,ch,:) = eval([Channels(ch).Name '(:,:)']);
        ChLabels{ch} = Channels(ch).Name;
    end
    
    ChannInfo = Channels;
    [~,tmpInd,~] = intersect(ChLabels,'photo');
    Segments(:,tmpInd,:) = [];
    ChLabels(tmpInd) = [];
    ChannInfo(tmpInd) = [];
end

%Changing Structure BCI raw structure:
function [Segments, Labels, Fs, ChannInfo] = ImportingBCIData(Filename)
    load(Filename);

    Fs = SampleRate;
    Segments = zeros(SegmentCount,ChannelCount,length(t));
    Labels = cell(ChannelCount,1);
    for ch = 1:ChannelCount
        for Seg = 1:SegmentCount
            Segments(Seg,ch,:) = eval([Channels(ch).Name '(Seg,:)']);
        end
        Labels{ch} = Channels(ch).Name;
    end
    ChannInfo = Channels;
    Segments(:,33:end,:) = [];
end

function [Dataset_Train,Labels_Train,Dataset_Test,Labels_Test] = DatasetGeneratorUnbal(Dataset, Labels,Kfold)
    %Labels are 0 or 1
    Labels_Train = Labels;
    Dataset_Train = Dataset;
    Dataset_Test = []; Labels_Test = [];
    TrainTestRatio = (1/Kfold);
    
    tmp1 = find(Labels_Train==1);
    tmp11 = randsample(tmp1,floor(length(tmp1)*TrainTestRatio));

    tmp0 = find(Labels_Train==0);
    tmp00 = randsample(tmp0,floor(length(tmp0)*TrainTestRatio));

    for t = 1:length(tmp11)
        Dataset_Test(end+1,:,:) = Dataset_Train(tmp11(t),:,:);
        Labels_Test(end+1,1) = Labels_Train(tmp11(t));
    end
    for t = 1:length(tmp00)
        Dataset_Test(end+1,:,:) = Dataset_Train(tmp00(t),:,:);
        Labels_Test(end+1,1) = Labels_Train(tmp00(t));
    end
    tmp01 = sort([tmp00;tmp11],'descend');
    for t = 1:length(tmp01)
        Dataset_Train(tmp01(t),:,:) = [];
        Labels_Train(tmp01(t)) = [];
    end

end

function [Dataset_Train,Labels_Train,Dataset_Test,Labels_Test] = DatasetGeneratorBal(Dataset, Labels,Kfold)
    %Labels are 0 or 1
    Labels_Train = Labels;
    Dataset_Train = Dataset;
    Dataset_Test = []; Labels_Test = [];
    TrainTestRatio = (1/Kfold);
    
    tmp1 = find(Labels_Train==1);
    tmp11 = randsample(tmp1,floor(length(tmp1)*TrainTestRatio));

    tmp0 = find(Labels_Train==0);
    tmp00 = randsample(tmp0,floor(length(tmp1)*TrainTestRatio));

    for t = 1:length(tmp11)
        Dataset_Test(end+1,:,:) = Dataset_Train(tmp11(t),:,:);
        Labels_Test(end+1,1) = Labels_Train(tmp11(t));
    end
    for t = 1:length(tmp00)
        Dataset_Test(end+1,:,:) = Dataset_Train(tmp00(t),:,:);
        Labels_Test(end+1,1) = Labels_Train(tmp00(t));
    end

    tmp01 = sort([tmp00;tmp11],'descend');
    for t = 1:length(tmp01)
        Dataset_Train(tmp01(t),:,:) = [];
        Labels_Train(tmp01(t)) = [];
    end

    tmp1 = find(Labels_Train==1);
    tmp00 = randsample(find(Labels_Train==0),length(find(Labels_Train==0))-length(tmp1));
    Dataset_Train(tmp00,:,:) = [];
    Labels_Train(tmp00) = [];
end

function [Dataset_Train1,Labels_Train1,Dataset_Test1,Labels_Test1] = PatientSpecDatasetsGenerator (PatientsDatasets, NumTestSample)

Dataset_Train = [];
AMIDataset_Train = [];
Labels_Train = [];
Dataset_Test1 = [];
AMIDataset_Test1 = [];
Labels_Test1 = [];

for s = 1:length(PatientsDatasets) 
    if s == NumTestSample
        Dataset_Test1 = PatientsDatasets{s}.Dataset;
        Labels_Test1 = PatientsDatasets{s}.Labels;
    else
        Dataset_Train = [Dataset_Train ; PatientsDatasets{s}.Dataset];
        Labels_Train = [Labels_Train ; PatientsDatasets{s}.Labels];
    end
end

p1 = randperm(length(Labels_Train)); % Shuffle the dataset
Labels_Train1 = Labels_Train(p1);
Dataset_Train1 = Dataset_Train(p1,:,:);

end

function [DistData] = DistGenerator(Data, DistMetricType)

    tmp = size(Data);
    DistData = zeros(tmp(1),tmp(2),tmp(2));

    for seg = 1:size(Data,1)
        tmp11(:,:) = Data(seg,:,:);
        D = pdist(tmp11, DistMetricType);
        DistData(seg,:,:) = squareform(D);
    end
end

function [AMIs] = AMIElec(Data,NLevel,MAXAMIK)
%[AMIs] = AMI(Data,NLevel,MAXAMIK)
%   Calculate the AMI by uniformly quantization. The Data should be a row
%   vector or a row matrix.
    
    Mins = min(Data(:));
    Maxs = max(Data(:));
    
    NormData = ((2.*Data)-(Maxs+Mins))./(Maxs-Mins);
    Data = NormData;
    Levels = [-1:((1)-(-1))/NLevel:1];
    
    NRow = size(Data,1);
    TotalSamp = size(Data,2);
    AMIs = zeros(NRow, MAXAMIK);
    JointProb = zeros(NLevel);
    Prob = zeros(1,NLevel);

    for m = 1:NLevel
       tmp = Data >= Levels(m) & Data < Levels(m+1);
       Prob(m) = sum(tmp(:))/length(Data(:));
    end

    Loc = cell(1,NLevel);
    for i = 1:NRow
        for m = 1:NLevel
            Loc{m} = find(Data(i,:) >= Levels(m) & Data(i,:) < Levels(m+1));
        end
        for k = 1:MAXAMIK
            for p = 1:NLevel
                for q = 1:NLevel
                    [Lia,~] = ismember(Loc{p},Loc{q}-k);
                    JointProb(p,q) = sum(Lia)/(TotalSamp-k);
                    if Prob(p)~= 0 && Prob(q)~=0 && JointProb(p,q)~=0
                        AMIs(i,k) = AMIs(i,k) + JointProb(p,q)*log2(JointProb(p,q)/(Prob(p)*Prob(q)));
                    end
                end
            end
        end
    end
end

function [PrePostAMIs] = AllAMIELECGenerator(Dataset_Train1, NLevel, MAXAMIK)

    PreAMIs = zeros(size(Dataset_Train1,1),size(Dataset_Train1,2)/2,MAXAMIK);
    PostAMIs = zeros(size(Dataset_Train1,1),size(Dataset_Train1,2)/2,MAXAMIK);
    PreDataset = Dataset_Train1(:,1:256,:);
    PostDataset = Dataset_Train1(:,257:end,:);
    parfor elec = 1:size(PreDataset,2)
        tmpDataElec = [];
        tmpDataElec(:,:) = PreDataset(:,elec,:);
        PreAMIs(:,elec,:) = AMIElec(tmpDataElec,NLevel,MAXAMIK);
    end
    parfor elec = 1:size(PostDataset,2)
        tmpDataElec = [];
        tmpDataElec(:,:) = PostDataset(:,elec,:);
        PostAMIs(:,elec,:) = AMIElec(tmpDataElec,NLevel,MAXAMIK);
    end
    PrePostAMIs = zeros(size(Dataset_Train1,1),size(Dataset_Train1,2),MAXAMIK);
    PrePostAMIs = [PreAMIs PostAMIs];

end

function [AMIs] = AMIXcrosElec(Data,NLevel)
% Data should contain samples in rows and electrodes in columns
    MAXAMIK = size(Data,2) - 1;

    Mins = min(Data(:));
    Maxs = max(Data(:));

    NormData = ((2.*Data)-(Maxs+Mins))./(Maxs-Mins);
    Data = NormData;
    Levels = [-1:((1)-(-1))/NLevel:1];

    NRow = size(Data,1);
    TotalSamp = size(Data,2);
    AMIs = zeros(NRow, MAXAMIK);
    JointProb = zeros(NLevel);
    Prob = zeros(1,NLevel);

    for m = 1:NLevel
       tmp = Data >= Levels(m) & Data < Levels(m+1);
       Prob(m) = sum(tmp(:))/length(Data(:));
    end

    Loc = cell(1,NLevel);
    for i = 1:NRow
        for m = 1:NLevel
            Loc{m} = find(Data(i,:) >= Levels(m) & Data(i,:) < Levels(m+1));
        end
        for k = 1:MAXAMIK
            for p = 1:NLevel
                for q = 1:NLevel
                    tmpq = Loc{q}-k;
%                     tmpq(tmpq<0) = tmpq(tmpq<0) + MAXAMIK;
                    [Lia,~] = ismember(Loc{p},tmpq);
                    JointProb(p,q) = sum(Lia)/(TotalSamp-k);
                    if Prob(p)~= 0 && Prob(q)~=0 && JointProb(p,q)~=0
                        AMIs(i,k) = AMIs(i,k) + JointProb(p,q)*log2(JointProb(p,q)/(Prob(p)*Prob(q)));
                    end
                end
            end
        end
    end
    AMIs = AMIs';
end
            
function [PrePostAMIs] = AllAMITimeGenerator(Dataset_Train1, NLevel)

    PreAMIs = zeros(size(Dataset_Train1,1),(size(Dataset_Train1,2)/2)-1,size(Dataset_Train1,3));
    PostAMIs = zeros(size(Dataset_Train1,1),(size(Dataset_Train1,2)/2)-1,size(Dataset_Train1,3));
    PreDataset = Dataset_Train1(:,1:256,:);
    PostDataset = Dataset_Train1(:,257:end,:);
    parfor seg = 1:size(PreDataset,1)
        tmpDataElec = [];
        tmpDataElec(:,:) = PreDataset(seg,:,:);
        PreAMIs(seg,:,:) = AMIXcrosElec(tmpDataElec,NLevel);
    end
    parfor seg = 1:size(PostDataset,1)
        tmpDataElec = [];
        tmpDataElec(:,:) = PostDataset(seg,:,:);
        PostAMIs(seg,:,:) = AMIXcrosElec(tmpDataElec,NLevel);
    end
    PrePostAMIs = zeros(size(Dataset_Train1,1),size(Dataset_Train1,2)-2,size(Dataset_Train1,3));
    PrePostAMIs = [PreAMIs PostAMIs];

end

% * 1) FreqBand initials
function FreqBand = FreqBand_init(varargin)
    %- MaxF is the Nyquist freq.
    
     if ~isempty(varargin) && isa(varargin{1},'double')
         MaxF = floor(varargin{1});
     else
         MaxF = 125;
     end
    
    FreqBand(1).Range = [0.5 4];
    FreqBand(1).Name = 'Delta';
    FreqBand(2).Range = [4 8];
    FreqBand(2).Name = 'Theta';
    FreqBand(3).Range = [8 14];
    FreqBand(3).Name = 'Alpha';
    FreqBand(4).Range = [14 30];
    FreqBand(4).Name = 'Beta';
    FreqBand(5).Range = [30 80];
    FreqBand(5).Name = 'Gamma';
    FreqBand(6).Range = [80 MaxF];
    FreqBand(6).Name = 'High-Gamma';
    FreqBand(7).Range = [0.5 8];
    FreqBand(7).Name = 'Delta-Theta'; 
    FreqBand(8).Range = [4 14];
    FreqBand(8).Name = 'Theta-Alpha';
    FreqBand(9).Range = [8 30];
    FreqBand(9).Name = 'Alpha-Beta';
    FreqBand(10).Range = [14 80];
    FreqBand(10).Name = 'Beta-Gamma';
    FreqBand(11).Range = [30 MaxF];
    FreqBand(11).Name = 'Gammas';
    FreqBand(12).Range = [0.5 MaxF];
    FreqBand(12).Name = 'All';
    
end

