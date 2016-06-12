function C = MatrixMul_Block(A, B, Block)
    if Block==0, Block=size(A,2); end
    C = zeros(size(A,1), size(B,2));
    n = 0;  N=size(A,2);
    while n < N
        idx1 = n+1;
        idx2 = n+Block;
        if idx2 >= N
            idx2 = N;
        end
        subA = A(:,idx1:idx2);
        subB = B(idx1:idx2,:);
        C = C + subA*subB;
        n = n + Block;
    end
