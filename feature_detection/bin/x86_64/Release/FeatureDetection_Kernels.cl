__constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
__constant sampler_t clamp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void feature_detection(__read_only image2d_t src, __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        float filterWeights[9*9] = 
        {
            1.234098e-04, 3.726653e-04, 9.118820e-04, 1.831564e-03, 2.865047e-03, 3.726653e-04, 9.118820e-04, 3.726653e-04, 1.234098e-04,
            3.726653e-04, 1.125351e-03, 2.753644e-03, 5.530844e-03, 8.636116e-03, 1.125351e-03, 2.753644e-03, 1.125351e-03, 3.726653e-04,
            9.118820e-04, 2.753644e-03, 6.737947e-03, 1.353353e-02, 2.113575e-02, 2.753644e-03, 6.737947e-03, 2.753644e-03, 9.118820e-04,
            1.831564e-03, 5.530844e-03, 1.353353e-02, 2.718281e-02, 4.247802e-02, 5.530844e-03, 1.353353e-02, 5.530844e-03, 1.831564e-03,
            2.865047e-03, 8.636116e-03, 2.113575e-02, 4.247802e-02, 6.635020e-02, 8.636116e-03, 2.113575e-02, 8.636116e-03, 2.865047e-03,
            1.831564e-03, 5.530844e-03, 1.353353e-02, 2.718281e-02, 4.247802e-02, 5.530844e-03, 1.353353e-02, 5.530844e-03, 1.831564e-03,
            9.118820e-04, 2.753644e-03, 6.737947e-03, 1.353353e-02, 2.113575e-02, 2.753644e-03, 6.737947e-03, 2.753644e-03, 9.118820e-04,
            3.726653e-04, 1.125351e-03, 2.753644e-03, 5.530844e-03, 8.636116e-03, 1.125351e-03, 2.753644e-03, 1.125351e-03, 3.726653e-04,
            1.234098e-04, 3.726653e-04, 9.118820e-04, 1.831564e-03, 2.865047e-03, 3.726653e-04, 9.118820e-04, 3.726653e-04, 1.234098e-04
        };
        float4 s = (float4)(0);
        int i1 = 0;
        int half_smoothing = 4;
        for (int y = -half_smoothing; y <= half_smoothing; ++y) {
            for (int x = -half_smoothing; x <= half_smoothing; ++x) {
                float4 cur_pix = convert_float4(read_imageui(src, clamp_sampler, coord + (int2)(x, y)));
                float4 gray_cur_pix = dot(cur_pix, (float4)(0.2126f, 0.7152f, 0.0722f, 0));
                s += filterWeights[i1] * gray_cur_pix;
                ++i1;
            }
        }
        if (s.x < 0) s = (float4)(0);
        write_imagef(dest, coord, s);
}


__kernel void feature_detection2 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    // float4 s = (float4)(0.0f);
    float4 Gx = (float4)(0);
    float4 Gy = (float4)(0);
    const float sobel_x[7][7] = {
        { 3,  2,  1,  0, -1, -2, -3},
        { 3,  2,  1,  0, -1, -2, -3},
        { 4,  3,  2,  0, -2, -3, -4},
        { 5,  3,  1,  0, -1, -3, -5},
        { 4,  3,  2,  0, -2, -3, -4},
        { 3,  2,  1,  0, -1, -2, -3},
        { 3,  2,  1,  0, -1, -2, -3}
    };

    const float sobel_y[7][7] = {
        { 3,  3,  4,  5,  4,  3,  3},
        { 2,  2,  3,  3,  3,  2,  2},
        { 1,  1,  2,  1,  2,  1,  1},
        { 0,  0,  0,  0,  0,  0,  0},
        {-1, -1, -2, -1, -2, -1, -1},
        {-2, -2, -3, -3, -3, -2, -2},
        {-3, -3, -4, -5, -4, -3, -3}
    };
    // Iterate over the 7x7 region
    for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
            float4 pixel = convert_float4(read_imagef(src, clamp_sampler, coord + (int2)(j, i)));
            Gx += pixel * sobel_x[i + 3][j + 3];
            Gy += pixel * sobel_y[i + 3][j + 3];
        }
    }
    float4 s = (float4)(fabs(Gx.x*Gx.x), fabs(Gy.x*Gy.x), fabs(Gx.x*Gy.x), 0);
    write_imagef(dest, coord, s);
}


__kernel void feature_detection22 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        const float4 in = read_imagef(src, clamp_sampler, coord);
    if((coord.x == 1000) && (coord.y==1000)) {printf("3");}
    float4 s = (float4)(0.0f);
    int i = 0;
    float filterWeights[5*5] =  {
								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f,
								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
								 0.0234375f, 0.0937500f, 0.1406250f, 0.0937500f, 0.0234375f,
								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f
							    };
    int half_smoothing = 2;
    for (int y = -half_smoothing; y <= half_smoothing; ++y) {
        for (int x = -half_smoothing; x <= half_smoothing; ++x) {
            float4 cur_pix = read_imagef(src, reflect_sampler, coord + (int2)(x,y));

            s += filterWeights[i] * cur_pix;
            ++i;
        }
    }
        float harris_k = 0.05;
        float4 r = (float4)(0);
        r.x = ((s.x * s.y - s.z * s.z) - harris_k * (s.x + s.y) * (s.x + s.y));
        if (r.x < 0) r.x = 0;  
        r.x = log10(fabs(r.x));
        if ((r.x < 10.65) || (r.x>10.8)) r.x=0;
        r.y = r.x;
        r.z = r.x;
        write_imagef(dest, coord, r);
    }

__kernel void feature_detection3 (__read_only image2d_t src_org, __read_only image2d_t src, __write_only image2d_t dest) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const uint4 org = read_imageui(src_org, clamp_sampler, pos);
    const float4 max = read_imagef(src, clamp_sampler, pos);
    int half_supr = 100;
    for (int y = -half_supr; y <= half_supr; y++) {
        for (int x = -half_supr; x <= half_supr; ++x) {
            const float4 surr = read_imagef(src, clamp_sampler, pos + (int2)(x, y));
            if (surr.x > max.x) {
                write_imageui(dest, pos, org);
                return;
            } else if (surr.x == max.x) {
                if (x < 0 && y < 0) {
                    write_imageui(dest, pos, org);
                    return;
                }
            }
        }
    }
    float4 d = (float4)(255.0f, 0.0f, 0.0f, 0.0f); 
    if (max.x != 0) {
        printf("(%d, %d) %f \n", pos.x, pos.y, max.x);
        for (int y = -10; y <= 10; y++) {
            for (int x = -10; x <= 10; ++x) {
                int2 draw_pos = pos + (int2)(x, y);
                write_imageui(dest, draw_pos, convert_uint4(d));
            }
        }
    }
}

