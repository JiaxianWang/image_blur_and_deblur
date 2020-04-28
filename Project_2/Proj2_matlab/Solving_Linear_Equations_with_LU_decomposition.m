function x = Solving_Linear_Equations_with_LU_decomposition(A, b)
% write your code here
% the output x should be inv(A)*b (or A\b), but you CANNOT use it as your final answer.
% you CANNOT use any high-level function in your code, for example inv(), matrix division, factorLU(), solve(), etc.
% function here function here function here function here function here
N = length(A);

% LU decomposition
% strating from gauss elimination for A to get U
% with permutation matrix P indicated the row swapping
P = eye(N);
% L matrix
L = eye(N);
% B matrix swapped
B_swapped = b;
% A matrix swapped, PA = A * Permutation
PA = A;
% gauss elimination in A to get U
U = A;
% swapped flag
swapped_flag = 0;
for i = 1:N
    if U(i,i) == 0
        %make sure U[i,i] != 0 and swap row in this case
        for j = i+1:N
            if U(j,i) ~= 0
                % indicated swapped 
                swapped_flag = 1;
                % swap row function here with permutation matrix
                temp_U = U(i,:);
                temp_P = P(i,:);
                
                
                U(i,:) = U(j,:);
                P(i,:) = P(j,:);
                
                
                U(j,:) = temp_U;
                P(j,:) = temp_P; 
                break
            end   
        end
    end
    for column = i+1:N
        if U(column,i) ~= 0
            % do subtraction here calculate fraction
            fraction = U(column,i) / U(i,i);
            
            % subtracte the row and get L through swapped matrix PA = LU
            L(column,i) = fraction;
            for k = 1:N
                U(column,k) = U(column,k) - fraction * U(i,k);
            end
        end
    end
end
PA = P*A;
B_swapped = P*B_swapped;
% recurtion when PA ~= A, indecated swapped
if swapped_flag == 1
    % CLEAN UP FOR MEMORY
    U = 0;L = 0;
    P = 0;X = 0;
    x = Solving_Linear_Equations_with_LU_decomposition(PA, B_swapped);
else
    

    % get swapped D matrix
    D = zeros(N,1);
    % since we know that L*D = B_swapped
    for i = 1:N
        D(i) = B_swapped(i) - d_sum(i,L,D);
    end

    % get X matrix sloven 
    % using UX = D
    X = zeros(N,1);

    for i = N:-1:1
        X(i) = (D(i) - x_sum(i,U,X,N)) / U(i,i); 
    end
    
    
    x = X;
end