clear all; clc; close all;                                               

vid=videoinput('winvideo',1);                                             %sets videoinput to the webcam, and the webcam device 1
                                                                           %displays the webcam input
figure(1);
set(vid,'ReturnedColorspace','rgb')
pause(2);                                                                 % pause 2 seconds before snapshot of background image
IM1=getsnapshot(vid);                                                     %get snapshot from the webcam video and store to IM1 variable
figure(1);subplot(3,3,1);imshow(IM1);title('Background'); 
final=0;
s=[];
pk1=0;
pk2=0;
for j=1:2
sum=0;%open up a figure and show the image stored in IM1 variable

for i=1:3
pause(4);                                                                 %pause a second before taking the test image snapshot
IM2=getsnapshot(vid);                                                     %get snapshot of test image and store to variable IM2
figure(1);subplot(3,3,(i+1));imshow(IM2);title('Gesture');                    %open up a figure and show the image stored in IM2 variable


IM3 = IM2 - IM1;                                                            %subtract Backround from Image
%figure(1);subplot(3,3,3);imshow(IM3);title('Subtracted');                   %show the subtracted image
IM3 = rgb2gray(IM3);                                                        %Converts RGB to Gray
%figure(1);subplot(3,3,4);imshow(IM3);title('Grayscale');                    %Display Gray Image

lvl = graythresh(IM3);
%text(0,350,['lvl: ' sprintf('%f',lvl)],'color','r','Fontsize',8);
IM3 = imbinarize(IM3,lvl);

%figure(1);subplot(3,3,5);imshow(IM3);title('Black&White'); %find the threshold value using Otsu's method for black and white
%thresh = multithresh(IM3,1);     
%IM3 = imquantize(IM3,thresh);%Converts image to BW, pixels with value higher than threshold value is changed to 1, lower changed to 0
                 %display black and white image
IM3 = bwareaopen(IM3, 10000);
IM3 = imfill(IM3,'holes');
%figure(1);subplot(3,3,6);imshow(IM3);title('Small Areas removed & Holes Filled');  
IM3 = imerode(IM3,strel('disk',15));                                        %erode image
IM3 = imdilate(IM3,strel('disk',20));                                       %dilate iamge
IM3 = medfilt2(IM3, [5 5]);                                                 %median filtering
%figure(1);subplot(3,3,7);imshow(IM3);title('Eroded,Dilated & Median Filtered');  
IM3 = bwareaopen(IM3, 10000);                                               %finds objects, noise or regions with pixel area lower than 10,000 and removes them
%figure(1);subplot(3,3,8);imshow(IM3);title('Processed');                    %displays image with reduced noise
IM3 = flipdim(IM3,1);                                                       %flip image rows
%figure(1);subplot(3,3,9);imshow(IM3);title('Flip Image');  
left = IM3(:, 1:end/2, :);
right = IM3(:, end/2+1:end, :);
%figure(2);
%figure(2);subplot(2,2,1);imshow(left);title('lefthalf');

REG=regionprops(left,'all');                                                %calculate the properties of regions for objects found 

CEN = cat(1, REG.Centroid);                                                  %calculate Centroid
[=B, L, N, A] = bwboundaries(left,'noholes');                                 %returns the number of objects (N), adjacency matrix A, object boundaries B, nonnegative integers of contiguous regions L

RND1=0;                                                                    % set variable RND to zero; to prevent errors if no object detected
pk=[];

%calculate the properties of regions for objects found
    for k =1:length(B)                                                      %for the given object k
            PER = REG(k).Perimeter;                                         %Perimeter is set as perimeter calculated by region properties 
            ARE = REG(k).Area;                                              %Area is set as area calculated by region properties
            RND1 = (4*pi*ARE)/(PER^2);                                       %Roundness value is calculated
            
            BND = B{k};                                                     %boundary set for object
            BNDx = BND(:,2);                                                %Boundary x coord
            BNDy = BND(:,1);                                                %Boundary y coord
            
            pkoffset = CEN(:,2)+.5*(CEN(:,2));                             %Calculate peak offset point from centroid
            [pks,locs] = findpeaks(BNDy,'minpeakheight',pkoffset);         %find peaks in the boundary in y axis with a minimum height greater than the peak offset
            pkNo1 = size(pks,1);                                            %finds the peak Nos
                                                                           %puts the peakNo in a string
         
    
    end
    
      
     if RND1==0                                                                     
      pkNo1=0;
                                                                              
    

 end

%figure(2);subplot(2,2,2);imshow(right);title('righthalf');
REG=regionprops(right,'all');                                                 %calculate the properties of regions for objects found 
CEN = cat(1, REG.Centroid);                                                 %calculate Centroid
[B, L, N, A] = bwboundaries(right,'noholes');                                 %returns the number of objects (N), adjacency matrix A, object boundaries B, nonnegative integers of contiguous regions L

RND2 = 0;                                                                    % set variable RND to zero; to prevent errors if no object detected

  
%calculate the properties of regions for objects found
    for k =1:length(B)                                                      %for the given object k
            PER = REG(k).Perimeter;                                         %Perimeter is set as perimeter calculated by region properties 
            ARE = REG(k).Area;                                              %Area is set as area calculated by region properties
            RND2 = (4*pi*ARE)/(PER^2);                                       %Roundness value is calculated
            
            BND = B{k};                                                     %boundary set for object
            BNDx = BND(:,2);                                                %Boundary x coord
            BNDy = BND(:,1);                                                %Boundary y coord
            
            pkoffset = CEN(:,2)+.5*(CEN(:,2));                             %Calculate peak offset point from centroid
            [pks,locs] = findpeaks(BNDy,'minpeakheight',pkoffset);         %find peaks in the boundary in y axis with a minimum height greater than the peak offset
            pkNo2 = size(pks,1); 
           
    
    end                                                                    % roundness is useful, for an object of same shape ratio, regardless of
       %text(0,350,['RND2: ' sprintf('%f',RND2)],'color','r','Fontsize',8);
      if RND2==0                                                                    % radius 5pixels will have the same roundness as a circle with radius
       pkNo2=0; 
      
      end

 pk(i)=pkNo1+pkNo2;
 
 figure(3);
 figure(3);subplot(1,1,1);imshow(IM3);title('resultant');
 text(0,250,['PKS: ' sprintf('%d',pk(i))],'color','r','Fontsize',20);
 %text(0,300,['RND1: ' sprintf('%d',RND1)],'color','r','Fontsize',20);
 %text(0,350,['RND2: ' sprintf('%d',RND2)],'color','r','Fontsize',20);


NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
pk1=num2str(pk(i));


text(0,450,['pk: ' sprintf('%d',pk(i))],'color','r','Fontsize',20);
%if pk(i)==0
%sum=sum*10;

%text(0,380,['SUM: ' sprintf('%d',sum)],'color','r','Fontsize',20);
%else
sum=sum+(pk(i)*10^(i-1));
%text(0,400,['SUM: ' sprintf('%d',sum)],'color','r','Fontsize',20);
%end


end
sum=str2num(fliplr(num2str(sum)));
s(j)=sum;
final=final+s(j);
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
pk1=num2str(sum);
Speak(obj,pk1);
%text(0,250,['PKS: ' sprintf('%d',sum)],'color','r','Fontsize',20);
end
NET.addAssembly('System.Speech');
obj = System.Speech.Synthesis.SpeechSynthesizer;
obj.Volume = 100;
pk1=num2str(final);
Speak(obj,pk1);
%text(0,250,['PKS: ' sprintf('%d',sum)],'color','r','Fontsize',20);