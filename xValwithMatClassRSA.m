clear; %Clears stored variables from Matlab
%%Configuration of Input
DatasetFold = ''; %Dataset Folder Address
BaselineDuration = 0.2; % Baseline correction (in seconds)   
Paradigms = {};
Paradigms(1).Tag = 'F'; %Paradigms of the study (do not use T and NT as we want to compare FT to FNT not FT to ZT)
Paradigms(2).Tag = 'Z'; %Add more paradigms after this if needed by continuing the pattern (i.e. Paradigms(3).Tag = 'A';)
RawEpochsChannels = 1:32; %Sets electrodes to use
%% Configuration of Output
kfolds = 10; %Defines the number of folds (k)
classifier = 'LDA'; %Defines which classifier to use
allAverages = [1:10]; %The seperate trial averages to calculate. Must be in increments of 1 for code to work as expected.
XLOutputName = ['k=10 classifier=' classifier datestr(now,'mmmm-dd-yyyy_HH_MM_SS') 'accuracies.xlsx'];
sheetsUsed = string({'TrialAveraging','ConfusionMatrix','PositivityNegativity'}); %#ok<STRCLQT> Sheet names for exported Excel workbook
negPosUsed = string({'TPR','TNR','FPR','FNR'}); %#ok<STRCLQT> Used to labe output from math applied to confusion matrix fro TPR, etc.

for trialAverage = allAverages
    accuracies_stored = [];
    accuracies_stored_line = 1; %The line number of the array for storage
    column = string(char('B' + trialAverage)); %Will break if averaging more than 23. Sets column by going the number of trials being averaged past B (e.g. for the trial average of 2, the column is 2 past B so it is D)
    for pt = 1:18 %Participant numbers to use
        if pt == 4 || pt == 5 %Skips participants 4 and 5
            continue;
        end
        for o = 1:length(Paradigms)
            if trialAverage == allAverages(1) %Adds patient number and paradigm only on the necessary rows during the first trial averaging
                for sheet = sheetsUsed
                    writecell({'PT','Paradigm'},XLOutputName,'Sheet',sheet,'Range','A1'); %Adds headers
                    writematrix(pt,XLOutputName,'Sheet',sheet,'Range',['A',num2str(accuracies_stored_line+1)]); %Writes patient number in first column
                    writematrix(string(Paradigms(o).Tag),XLOutputName,'Sheet',sheet,'Range',['B',num2str(accuracies_stored_line+1)]); %Writes paradigm in second column
                end
            end
            tmpfileAdd = [DatasetFold 'P' num2str(pt) '_CVI_Artifact_Rejection_' Paradigms(o).Tag 'T.mat']; %Filenames to import. Be sure to match the formating of your filenames
            [PtSegments{pt,1}, ChLabels, Fs, ~] = ImportingBCIData(tmpfileAdd);
            tmpfileAdd = [DatasetFold 'P' num2str(pt) '_CVI_Artifact_Rejection_' Paradigms(o).Tag 'NT.mat'];
            [PtSegments{pt,2}, ChLabels, Fs, ChannInfo] = ImportingBCIData(tmpfileAdd);

            Paradigms(o).RawEpochFeatures = zeros(size(PtSegments{pt,1},1)+size(PtSegments{pt,2},1),length(RawEpochsChannels)*size(PtSegments{pt,1}(:,:,BaselineDuration*Fs+1:end),3)); %Creates array of zeroes of appropriate dimensions for the data window sample, correcting for baseleine duration

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
        Paradigms(o).SegmentsLabels = [ones(size(PtSegments{pt,1},1),1) ; zeros(size(PtSegments{pt,2},1),1)]; %Creates list of 1's for targets and 0's for nontargets
        [CM, accuracy, predY, pVal, classifierInfo] = classifyEEG (Paradigms(o).RawEpochFeatures, Paradigms(o).SegmentsLabels, 'averageTrials', trialAverage, 'classify', classifier, 'nFolds', kfolds); %Does cross-validation
        accuracies_stored(accuracies_stored_line,1) = accuracy;
        accuracies_stored_line = accuracies_stored_line + 1;
        writecell({strjoin(string(CM))},XLOutputName,'Sheet',sheetsUsed(2),'Range',column+accuracies_stored_line);%Writes confusion matrix to the sheet
        end
    end    
    writecell({string(['Avg Accuracy: ' num2str(trialAverage) ' trials'])},XLOutputName,'Sheet',sheetsUsed(1),'Range',column+'1'); %Header for columns of data
    writecell({string(['TP,FN,FP,TN:' num2str(trialAverage) ' trials'])},XLOutputName,'Sheet',sheetsUsed(2),'Range',column+'1');
    writematrix(accuracies_stored,XLOutputName,'Sheet',sheetsUsed(1),'Range',column+'2');
end
outputedCMs = readcell(XLOutputName,'Sheet',sheetsUsed(2),'Range','C2'); %Reads in all the confusion matrices
newCM = [];
for n = 1:length(allAverages)
    for q = 1:length(outputedCMs)
        myVal = str2double(strsplit(outputedCMs{q,n})); %Parses each confusion matrix into the 4 numbers (True Positives, False Negatives, False Positives, True Negatives)
        TPR = myVal(1)/(myVal(1)+myVal(2)+myVal(3)+myVal(4)); %Calcuates the rate of each (True positivity rate, etc.)
        TNR = myVal(4)/(myVal(1)+myVal(2)+myVal(3)+myVal(4));
        FPR = myVal(3)/(myVal(1)+myVal(2)+myVal(3)+myVal(4));
        FNR = myVal(2)/(myVal(1)+myVal(2)+myVal(3)+myVal(4));
        newCM{q,n} = [TPR,TNR,FPR,FNR]; %Stores the 4 values in array
    end
end
writecell(newCM,XLOutputName,'Sheet',sheetsUsed(3),'Range','C2'); %Writes the rates to the output sheet
index = 1;
for c = 1:length(allAverages)
    for i = 1:4
        negPosLabels{1,index} = (strjoin([negPosUsed(i), ': ',num2str(c), ' trials'])); %Stores headers in array
        index = index + 1;
    end
end
writecell(negPosLabels,XLOutputName,'Sheet',sheetsUsed(3),'Range','C1'); %Writes headers to Negativity Positivity sheet
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