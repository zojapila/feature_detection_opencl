__constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
__constant sampler_t clamp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


// __kernel void feature_detection (
//     __read_only image2d_t src,
//     __write_only image2d_t dest) {
//         int2 coord = (int2)(get_global_id(0), get_global_id(1));
//         // uint width = get_global_size(0);
// 	    // uint height = get_global_size(1);
//         float4 s = (float4)(0.0f);

// 	    float4 Gx = (float4)(0);
// 	    float4 Gy = Gx;
//         //sobel reference app
//         // pixel = dot(convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
//         float4 i00 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i10 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i20 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i01 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i11 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i21 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i02 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i12 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
//         float4 i22 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));

// 		Gx =   (i00 + (float4)(2) * i10 + i20 - i02  - (float4)(2) * i12 - i22);

// 		Gy =   (i00 - i20  + (float4)(2)*i01 - (float4)(2)*i21 + i02  -  i22);

// 		// Gx = native_divide(native_sqrt(Gx * Gx + Gy * Gy), (float4)(2));

//             s.x = fabs(Gx.x *Gx.x ) ;
//             s.y = fabs(Gy.x *Gy.x ) ;
//             s.z = fabs(Gx.x *Gy.x) ;

//         // write_imageui(dest, coord, s);
//         write_imagef(dest, coord, s);

//     }
__kernel void feature_detection (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        const float4 in = convert_float4(read_imageui(src, clamp_sampler, coord));
        float4 pixel = dot(in, (float4)(0.2126f, 0.7152f, 0.0722f, 0)); 
        float minVal = pixel.x; // Initial high value for min operation
        if((coord.x == 1000) && (coord.y==1000)) {printf("1");}
        
        int er_size = 10;

    // // Iterate over the 5x5 region
    for (int i = -er_size; i <= er_size; i++) {
        for (int j = -er_size; j <= er_size; j++) {
            // uint4 pixel = read_imageui(src, clamp_sampler, coord + (int2)(j, i));
            float4 pixel = convert_float4(read_imageui(src, clamp_sampler, coord + (int2)(i, j)));
            pixel = dot(pixel, (float4)(0.2126f, 0.7152f, 0.0722f, 0)); 
            minVal = fmax(minVal, pixel.x); // Update minimum value
        }
    }
    // write_imagef(dest, coord, convert_float4(pixel));
    // printf("%f \n", minVal);
    write_imagef(dest, coord, (float4)(minVal));
}
        // write_imageui(dest, coord, convert_uint4(in));


__kernel void feature_detection2 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 s = (float4)(0.0f);
    const float4 in = convert_float4(read_imageui(src, clamp_sampler, coord));
    if((coord.x == 1000) && (coord.y==1000)) {printf("2");}
    float4 Gx = (float4)(0);
    float4 Gy = (float4)(0);

    // Define the Sobel 7x7 kernel
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
            pixel = dot(pixel, (float4)(0.2126f, 0.7152f, 0.0722f, 0)); 
             // Convert to grayscale

            Gx += pixel * sobel_x[i + 3][j + 3];
            Gy += pixel * sobel_y[i + 3][j + 3];
        }
    }

    // s.x = sqrt(Gx.x * Gx.x + Gy.x * Gy.x);
    // s.y = sqrt(Gx.y * Gx.y + Gy.y * Gy.y);
    // s.z = sqrt(Gx.z * Gx.z + Gy.z * Gy.z);
    // s.w = 1.0f; // Optional alpha channel
    s.x = fabs(Gx.x *Gx.x ) ;
    s.y = fabs(Gy.x *Gy.x ) ;
    s.z = fabs(Gx.x *Gy.x) ;

    write_imagef(dest, coord, s);
}

// __kernel void feature_detection (
//     __read_only image2d_t src,
//     __write_only image2d_t dest) {
//     int2 coord = (int2)(get_global_id(0), get_global_id(1));
//     float4 s = (float4)(0.0f);

    // }
    //  const float4 max = read_imagef(src, clamp_sampler, pos);

    // if (max.x < threshold) {
    //     write_imagef(dest, pos, (float4)0.0f);
//         { 2,  1,  0, -1, -2},
//         { 2,  1,  0, -1, -2},
//         { 4,  2,  0, -2, -4},
//         { 2,  1,  0, -1, -2},
//         { 2,  1,  0, -1, -2}
//     };

//     const float sobel_y[5][5] = {
//         { 2,  2,  4,  2,  2},
//         { 1,  1,  2,  1,  1},
//         { 0,  0,  0,  0,  0},
//         {-1, -1, -2, -1, -1},
//         {-2, -2, -4, -2, -2}
//     };

//     // Iterate over the 5x5 region
//     for (int i = -2; i <= 2; i++) {
//         for (int j = -2; j <= 2; j++) {
//             float4 pixel = convert_float4(read_imageui(src, reflect_sampler, coord + (int2)(j, i)));
//             pixel = dot(pixel, (float4)(0.2126f, 0.7152f, 0.0722f, 0));  // Convert to grayscale

//             Gx += pixel * sobel_x[i + 2][j + 2];
//             Gy += pixel * sobel_y[i + 2][j + 2];
//         }
//     }
//     s.x = fabs(Gx.x *Gx.x ) ;
//     s.y = fabs(Gy.x *Gy.x ) ;
//     s.z = fabs(Gx.x *Gy.x) ;
//     // s.x = sqrt(Gx.x * Gx.x + Gy.x * Gy.x);
//     // s.y = sqrt(Gx.y * Gx.y + Gy.y * Gy.y);
//     // s.z = sqrt(Gx.z * Gx.z + Gy.z * Gy.z);


//     write_imagef(dest, coord, s);
// }


__kernel void feature_detection22 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        const float4 in = read_imagef(src, clamp_sampler, coord);
    // printf("3");
    if((coord.x == 1000) && (coord.y==1000)) {printf("3");}

    //SMOOTHING
    float4 s = (float4)(0.0f);
    int i = 0;
    // //Half smoothing smoothing size / 2 (5/2)
    // }
    //  const float4 max = read_imagef(src, clamp_sampler, pos);

    // if (max.x < threshold) {
    //     write_imagef(dest, pos, (float4)0.0f);

    // //filterWeights is gaussianfilter
    float filterWeights[5*5] = 
							{
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
        r.x = ((s.x * s.y - s.z * s.z) - harris_k * (s.x + s.y) * (s.x + s.y))/ 7000000000;
        // printf("%f",r.x);
    // }
    //  const float4 max = read_imagef(src, clamp_sampler, pos);

    // if (max.x < threshold) {
    //     write_imagef(dest, pos, (float4)0.0f);
        if (r.x < 200) {
            r.x=0;
        }
        // else {
        //     printf("%f \n", r.x);
        // }
        r.y = r.x;
        r.z = r.x;


        write_imagef(dest, coord, r);
    }

__kernel void feature_detection3 (
    __read_only image2d_t src_org,
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 pos = (int2)(get_global_id(0), get_global_id(1));
    float4 minVal = (float4)(1.0f); // Initial high value for min operation
    const uint4 org = read_imageui(src_org, clamp_sampler, pos);
    const float4 in = convert_float4(read_imageui(src, clamp_sampler, pos));

    // printf("4");
    if((pos.x == 1000) && (pos.y==1000)) {printf("4");}

    // // // Iterate over the 5x5 region
    // for (int i = -1; i <= 1; i++) {
    //     for (int j = -1; j <= 1; j++) {
    //         float4 pixel = read_imagef(src, clamp_sampler, coord + (int2)(j, i));
    //         minVal = fmin(minVal, pixel); // Update minimum value
    //     }
    // }
     const float4 max = read_imagef(src, clamp_sampler, pos);

    // if (max.x < threshold) {
    //     write_imagef(dest, pos, (float4)0.0f);
    //     return;
    // }
    int half_supr = 30;

    for (int y = -half_supr; y <= half_supr; y++) {
        for (int x = -half_supr; x <= half_supr; ++x) {
            const float4 r = read_imagef(src, reflect_sampler, pos + (int2)(x,y));
            if (r.x > max.x) {
                write_imagef(dest, pos, convert_float4(org));
                return;
            }
            // else
            // {
            //     write_imagef(dest, pos, convert_float4(org));
            //     // return;
            // }

        }

    }
    float4 d = (255, 0, 0, 0);
    // if (max.x > 0) {
    // // printf("ID: ( {%d, %d, %f)", pos.x, pos.y, max.x);
    // }
    write_imagef(dest, pos, d);

    // write_imagef(dest, coord, in);
    }


__kernel void feature_detection4 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        // printf("5");
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        if((coord.x == 1000) && (coord.y==1000)) {printf("5");}
        const float4 in = read_imagef(src, clamp_sampler, coord);
        write_imageui(dest, coord, convert_uint4(in));
    }