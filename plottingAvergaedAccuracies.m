tableName = 'acc'; %Name of the imported CSV file
paradigm1 = 'Functional'; %Label for column 1
paradigm2 = 'Zoom'; %Label for column 2
maximumTrialsAveraged = 6; %The maximum x axis value, should be equal to the largest number of trials averaged
lineThickness = 3.5; %Weight of line
markerSize = 20; %Size of data marker symbol
markerCharacter = 'x'; %Uses x as data marker symbol on line
xAxisLimit = [0.9 maximumTrialsAveraged+0.1]; %Minimum and maximum values on axis, good to use 0.1 less than minimum and 0.2 more than maximum trials averaged
yAxisLimit = [82 94]; %Set bounds for y axis, trim to better show data range

p1 = plot(eval(append(tableName,'.',paradigm1)),'Color', 'k', 'LineWidth',lineThickness,'MarkerSize',markerSize,Marker=markerCharacter); %Plots paradigm 1
hold on %Keeps plot open for all the following additions and modifications
p2 = plot(eval(append(tableName,'.',paradigm2)),'--','Color','k','LineWidth',lineThickness,'MarkerSize',markerSize,Marker=markerCharacter); %Plots paradigm 2, Delete '--' to make a solid line, change color from 'k' to change from black
xlim(xAxisLimit) %Sets minimum and maximum values on x axis
ylim(yAxisLimit) %Sets minimum and maximum values on y axis
interval_x_axis=1;
set(gca,'XTick',1:interval_x_axis:maximumTrialsAveraged) %Forces x axis labels in increments of 1 to match the trials used
set(gca,'fontsize', 20) %Font for numbers on axes
[hleg1, hobj1] = legend(paradigm1,paradigm2,'Box','off'); %creates legend with no surrounding box
ylabel('Accuracy (%)') %Labels y axis
xlabel('Number of Trials Averaged') %Labels x axis
set(hleg1,'position',[0.72 0.8 0.3 0.1]) %Relative legend location ([left bottom width height])
