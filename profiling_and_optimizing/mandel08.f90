subroutine mandel_set8(xmin, xmax, ymin, ymax, width, height, itermax, n)
    real(8), intent(in)   :: xmin, xmax, ymin, ymax
    integer, intent(in)   :: width, height, itermax
    integer               :: niter
    integer, dimension(height, width), intent(out) :: n
    integer               :: x, y
    real(8)               :: xstep, ystep
    
    xstep = (xmax - xmin) / (width - 1)
    ystep = (ymax - ymin) / (width - 1)
    do x = 1, width
        do y = 1, height
            call mandel8(xmin + (x - 1) * xstep, ymin + (y - 1) * ystep, itermax, niter)
            n(y, x) = niter
        end do
    end do
end subroutine mandel_set8

subroutine mandel8(cre, cim, itermax, n)
    real(8), intent(in)      :: cre, cim
    integer, intent(in)      :: itermax
    integer, intent(out)     :: n
    real(8)                  :: re2, im2, re, im

    re = cre
    im = cim 
    do n = 0, itermax - 1
        re2 = re ** 2
        im2 = im ** 2
        if (re2 + im2 > 4.0) then
            exit
        end if
        im = 2 * re * im + cim
        re = re2 - im2 + cre
    end do
    if (n == itermax) n = 0
end subroutine mandel8
        

