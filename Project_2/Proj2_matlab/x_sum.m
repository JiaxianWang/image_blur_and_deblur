function sum = x_sum(row,upper,X,N)
sum_x = 0;
for r = 1:(N-row)
    sum_x = sum_x + upper(row,N-r+1) * X(N-r+1);
end
sum = sum_x;
end