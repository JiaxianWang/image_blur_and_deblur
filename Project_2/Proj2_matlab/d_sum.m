function sum = d_sum(row,lower,D)
sum_d = 0;
for r = 1:(row-1)
    sum_d = sum_d + lower(row,r)*D(r);
end
sum = sum_d;
end