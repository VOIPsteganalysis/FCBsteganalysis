function F = MTJCE(FCB)
% -------------------------------------------------------------------------
% 2019.1.7
% 
% -------------------------------------------------------------------------
%
% -------------------------------------------------------------------------
% Input:  FCB ... path to the FCB
%
% Output: f ....... extracted Miao features
% 
% -------------------------------------------------------------------------

	matrix1 = single(FCB);
	matrix2 = matrix1;
	matrix2 = floor(matrix2/5);
	matrix = int8(matrix2);
	F = zeros(1,66);
	
	track_0=matrix(:,[1,6]);
	track_1=matrix(:,[2,7]);
	track_2=matrix(:,[3,8]);
	track_3=matrix(:,[4,9]);
	track_4=matrix(:,[5,10]);
    feature1 = [];
	feature1 = [feature1;getTPM(track_0)];
	feature1 = [feature1;getTPM(track_1)];
	feature1 = [feature1;getTPM(track_2)];
	feature1 = [feature1;getTPM(track_3)];
	feature1 = [feature1;getTPM(track_4)];
	
	feature1 = double(mean(feature1)); 
	
	feature2 = [];
	feature2 = [feature2;getEntropy(track_0)];
	feature2 = [feature2;getEntropy(track_1)];
	feature2 = [feature2;getEntropy(track_2)];
	feature2 = [feature2;getEntropy(track_3)];
	feature2 = [feature2;getEntropy(track_4)];
	
	feature2 = double( mean( feature2)); 	
	F = [feature1,feature2];

function F = getTPM(track)
% get transition probability matrix A1 --> A2
F = zeros(8,8);
A1 = track(:,1)+1;
A2 = track(:,2)+1;
dn = max(hist(A1(:), 1:9), 1);% normalization factors

for i = 1:8
    FF = A2(A1 == i); % filtered version
    if ~isempty(FF)
		for j = 1:8
			F(i, j) = nnz(FF==j) / dn(i);
		end
	end
end
F = F(:)';

function F = getEntropy(track)

F = zeros(1,2); 
A1 = track(:,1);
A2 = track(:,2);

[m,n]=size(A1);     
a = A1+1;
b = A2+1;
Joint_H=zeros(8,8);

for i = 1:m
    for j = 1:n
        index_x=a(i,j);
        index_y=b(i,j);
        Joint_H(index_x,index_y)=Joint_H(index_x,index_y)+1;
    end
end

Joint_p = Joint_H./(m*n);
en_j = 0.0;
for i = 1:8
    for j = 1:8
        if Joint_p(i,j)~=0
           en_j = en_j+Joint_p(i,j)*log2( Joint_p(i,j));
        end
    end
end

F(1) = -en_j;

Pulse1_H=zeros(1,8);
for i = 1:m
    for j = 1:n
        index_x=a(i,j);       
        Pulse1_H(1,index_x)=Pulse1_H(1,index_x)+1;
    end
end

Pulse1_p = Pulse1_H./(m*n);
en_c = 0.0;
for i = 1:8
        if Pulse1_p(i)~=0
           en_c = en_c+Pulse1_p(i)*log2(Pulse1_p(i));
        end
end
F(2) = -en_c;
F(2) = F(1)-F(2);