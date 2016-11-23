

[num,txt,raw]=xlsread('./id_label_val.csv'); %
[outnum,outtxt,outraw]=xlsread('./finalOutputValidation.csv'); %

tp=0; tn=0; fp=0; fn=0;

length(find(outnum(5:36,3)==1))
ids=unique(num(:,1));
prediction_time=zeros(1,1);
k=1;
for i=1:length(ids)
    start=find(outnum(:,1)==ids(i), 1 );
    stop=find(outnum(:,1)==ids(i), 1, 'last' );
    if(~isempty(find(outnum(start:stop,3)==1)))
        if(num(i,2)==1) %meaning both are 1
            tp=tp+1;
        else
            fp=fp+1;
        end
        prediction_time(1,k)=(outnum(stop,2)-outnum(start,2))/3600;
        k=k+1;
    else %it is 0
        if(num(i,2)==0) %meaning both are 1
            tn=tn+1;
        else
            fn=fn+1;
        end
    end
end

Sensitivity_score=tp/(tp+fn);

specificity=tn/(tn+fp);
Specificity_score=(specificity - 0.99)*100;

prediction_time=sort(prediction_time);
M=median(prediction_time);
if(M<72)
    median_pred_time_clipped_at_72=M;
else
    median_pred_time_clipped_at_72=72;
end

Median_pred_time_score = median_pred_time_clipped_at_72 / 72;


Final_score = 0.75*Sensitivity_score + 0.2*Median_pred_time_score + 0.05*Specificity_score;
disp(['The final Score is: ', num2str(Final_score)]);
disp(['Sensitivity: ', num2str(Sensitivity_score)]);
disp(['specificity: ', num2str(specificity)]);
