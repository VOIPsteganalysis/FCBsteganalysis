function F = NJPB(FCB)

% feature part
matrix1 = single(FCB);

% PreProcess
matrix2 = matrix1;
matrix2 = floor(matrix2/5);
matrix = int8(matrix2);
X = matrix;
F1 = ExtractIntrablock(X); 
F2 = ExtractInterblock(X); 
F3 = ExtractIntraDifblock(X); 
F4 = ExtractInterDifblock(X); 
F =[F1;F2;F3;F4];

%% Intrablock
function F = ExtractIntrablock(A)
T = 7;
F = zeros(64,1); % Intrablock part;
%% horizontal

F1h = extractNB( A(:,1:end-1), A(:,2:end), T);
F1h = normalize( reshape(F1h,[],1));

%% vertical
F1v = extractNB( A(1:end-1,:), A(2:end,:), T);
F1v = normalize( reshape(F1v,[],1));

%% diagonal
F1d = extractNB( A(1:end-1,1:end-1), A(2:end,2:end), T);
F1d = normalize( reshape(F1d,[],1));

%% minor diagonal
F1m = extractNB( A(2:end,1:end-1), A(1:end-1,2:end), T);
F1m = normalize( reshape(F1m,[],1));

F = (F1h+F1v+F1d+F1m)/4;

 
%% Interblock    
function F = ExtractInterblock(A)
T = 7; 
F = zeros(64,1); % inter-block part; 
for MODE = 1:20

    AA = PlaneToVecMode(A,MODE);
    
    % horizontal
    F2h = extractNB( AA(:,1:end-1), AA(:,2:end),T);
    F2h = normalize( reshape(F2h,[],1));

    % vertical
    F2v = extractNB( AA(1:end-1,:), AA(2:end,:),T); 
    F2v = normalize( reshape(F2v,[],1));

    %% diagonal
    F2d = extractNB( AA(1:end-1,1:end-1), AA(2:end,2:end), T);
    F2d = normalize( reshape(F2d,[],1));

    %% minor diagonal
    F2m = extractNB( AA(2:end,1:end-1), AA(1:end-1,2:end), T);
    F2m = normalize( reshape(F2m,[],1));

    F = F+(F2h+F2v+F2d+F2m)/4;
end
F = F/20;

function F = ExtractIntraDifblock(A)

T = 7;
F = zeros(64,1); % Intrablock part;
%% horizontal
%Ad = conv2(A,[-1 1],'valid');    % <=> A(:,1:end-1) - A(:,2:end);
Ad = A(:,1:end-1) - A(:,2:end);
Ad = abs(Ad);
F1h = extractNB( Ad(:,1:end-1), Ad(:,2:end), T);
F1h = normalize( reshape(F1h,[],1));

%% vertical
%Ad = conv2(A,[-1;1],'valid');    % <=> A(1:end-1,:) - A(2:end,:);
Ad =  A(1:end-1,:) - A(2:end,:);
Ad = abs(Ad);
F1v = extractNB( Ad(1:end-1,:), Ad(2:end,:), T);
F1v = normalize( reshape(F1v,[],1));

%% diagonal
%Ad = conv2(A,[-1 0;0 1],'valid');% <=> A(1:end-1,1:end-1) - A(2:end,2:end);
Ad = A(1:end-1,1:end-1) - A(2:end,2:end);
Ad = abs(Ad);
F1d = extractNB( Ad(1:end-1,1:end-1), Ad(2:end,2:end), T);
F1d = normalize( reshape(F1d,[],1));

%% minor diagonal
%Ad = conv2(A,[0 1;-1 0],'valid');% <=> A(2:end,1:end-1) - A(1:end-1,2:end);
Ad = A(2:end,1:end-1) - A(1:end-1,2:end);
Ad = abs(Ad);
F1m = extractNB( Ad(2:end,1:end-1), Ad(1:end-1,2:end), T);
F1m = normalize( reshape(F1m,[],1));

F = (F1h+F1v+F1d+F1m)/4;

function F = ExtractInterDifblock(A)
T = 7; 
F = zeros(64,1); % inter-block part; 
for MODE = 1:20

    AA = PlaneToVecMode(A,MODE);
    
    %Ad = conv2(AA,[-1 1],'valid');   % <=> A(:,1:end-1) - A(:,2:end);
    Ad = A(:,1:end-1) - A(:,2:end);
    Ad = abs(Ad);
    % horizontal
    F2h = extractNB( Ad(:,1:end-1), Ad(:,2:end),T);
    F2h = normalize( reshape(F2h,[],1));

    % vertical
    Ad = A(1:end-1,:) - A(2:end,:);
    Ad = abs(Ad);
    F2v = extractNB( Ad(1:end-1,:), Ad(2:end,:),T); 
    F2v = normalize( reshape(F2v,[],1));

    %% diagonal
    %Ad = conv2(AA,[-1 0;0 1],'valid');% <=> A(1:end-1,1:end-1) - A(2:end,2:end);
    Ad = A(1:end-1,1:end-1) - A(2:end,2:end);
    Ad = abs(Ad);
    F2d = extractNB( Ad(1:end-1,1:end-1), Ad(2:end,2:end), T);
    F2d = normalize( reshape(F2d,[],1));

    %% minor diagonal
    %Ad = conv2(AA,[0 1;-1 0],'valid');% <=> A(2:end,1:end-1) - A(1:end-1,2:end);
    Ad = A(2:end,1:end-1) - A(1:end-1,2:end);
    Ad = abs(Ad);
    F2m = extractNB( Ad(2:end,1:end-1), Ad(1:end-1,2:end), T);
    F2m = normalize( reshape(F2m,[],1));

    F = F+(F2h+F2v+F2d+F2m)/4;

end
F = F/20;

function F = extractNB(A1,A2,t)

% 2nd order cooccurence
F = zeros(t+1,t+1);
for i=0:t
    FF = A2(A1==i);
    if ~isempty(FF)
        for j=0:t
            F(i+1,j+1) = sum(FF(:)==j);
        end
    end
end

function f = normalize(f)
S = sum(f(:));
if S~=0, f=f/S; end

function Mat=PlaneToVecMode(plane,MODE)
mask = reshape(1:20,4,5);
[i,j] = find(mask==MODE);
Mat = [plane(:,j),plane(:,j+5)];
