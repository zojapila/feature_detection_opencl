// __constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
// __constant sampler_t clamp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


// __kernel void feature_detection (
//     __read_only image2d_t src,
//     __write_only image2d_t dest) {


//     // ARG32TOFLOAT  
//     const int2 pos = {get_global_id(0), get_global_id(1)};
//     // const float4 in = read_imagef(src, clamp_sampler, pos);
//     // float4 pix = (float4)(in.x * 0.2989f + in.y * 0.5870f + in.z * 0.1140f);

//     //SMOOTHING
//     float4 sum = (float4)(0.0f);
//     int i = 0;
//     //Half smoothing smoothing size / 2 (5/2)

//     //filterWeights is gaussianfilter
//     float filterWeights[5*5] = 
// 							{
// 								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f,
// 								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
// 								 0.0234375f, 0.0937500f, 0.1406250f, 0.0937500f, 0.0234375f,
// 								 0.0156250f, 0.0625000f, 0.0937500f, 0.0625000f, 0.0156250f,
// 								 0.0039062f, 0.0156250f, 0.0234375f, 0.0156250f, 0.0039062f
// 							};
//     int half_smoothing = 2;
//     for (int y = -half_smoothing; y <= half_smoothing; ++y) {
//         for (int x = -half_smoothing; x <= half_smoothing; ++x) {
//             float4 cur_pix = read_imagef(src, reflect_sampler, pos + (int2)(x,y));
//             float4 cur_pix_gray = (float4)(cur_pix.x * 0.2989f + cur_pix.y * 0.5870f + cur_pix.z * 0.1140f);

//             sum += filterWeights[i] * cur_pix_gray;
//             ++i;
//         }
//     }
    
//     write_imagef(dest, pos, sum);

// }

// __kernel void feature_detection2 (
//     __read_only image2d_t src,
//     __write_only image2d_t dest) {
//         int2 coord = (int2)(get_global_id(0), get_global_id(1));
//         const float4 in = read_imagef(src, clamp_sampler, coord);
        
//         uint width = get_global_size(0);
// 	    uint height = get_global_size(1);
//         float4 s = (float4)(0.0f);

// 	    float4 Gx = (float4)(0);
// 	    float4 Gy = Gx;
//         //sobel reference app
//         float4 i00q
// 		Gx =   i00 + (float4)(2) * i10 + i20 -    write_imagef(dest, pos, sum);

// }
// 		// Gx = native_divide(native_sqrt(Gx * Gx + Gy * Gy), (float4)(2));

//             s.x = Gx.x *Gx.x ;
//             s.y = Gy.x *Gy.x ;
//             s.z = Gx.x *Gy.x;


//         float harris_k = 0.05;
//         float4 r = (float4)(0);
//         r.x = ((s.x * s.y - s.z * s.z) - harris_k * (s.x + s.y) * (s.x + s.y));
        
//         // r = fast_normalize(r);

//         // if (r.x< 10) {
//         //     r.x = 0;
//         // }
//         // r.y = r.x;
//         // r.z = r.x;
//         // write_imagef(dest, coord, Gx);
//         write_imageui(dest, coord, convert_uint4(s));
//         // }
//     }

// __kernel void feature_detection3 (
//     __read_only image2d_t src,
//     __global float* row_max_values) {

//     const int row = get_global_id(0);
//     const int width = get_image_width(src);
//     float row_max = 0.0f;

//     for (int x = 0; x < width; ++x) {
//         const int2 pos = {x, row};
//         const float4 s = read_imagef(src, reflect_sampler, pos);
//         row_max = max(s.x, row_max);
//     }

//     row_max_values[row] = row_max;
// }

__constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
__constant sampler_t clamp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void feature_detection (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        // uint width = get_global_size(0);
	    // uint height = get_global_size(1);
        float4 s = (float4)(0.0f);

	    float4 Gx = (float4)(0);
	    float4 Gy = Gx;
        //sobel reference app
        // pixel = dot(convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
        float4 i00 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i10 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i20 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y + 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i01 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i11 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i21 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y + 0))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i02 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 1, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i12 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x - 0, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));
        float4 i22 = dot(convert_float4(read_imageui(src, reflect_sampler, (int2)(coord.x + 1, coord.y - 1))), (float4)(0.2126f, 0.7152f, 0.0722f, 0));

		Gx =   (i00 + (float4)(2) * i10 + i20 - i02  - (float4)(2) * i12 - i22);

		Gy =   (i00 - i20  + (float4)(2)*i01 - (float4)(2)*i21 + i02  -  i22);

		// Gx = native_divide(native_sqrt(Gx * Gx + Gy * Gy), (float4)(2));

            s.x = fabs(Gx.x *Gx.x ) ;
            s.y = fabs(Gy.x *Gy.x ) ;
            s.z = fabs(Gx.x *Gy.x) ;

        // write_imageui(dest, coord, s);
        write_imagef(dest, coord, s);

    }


__kernel void feature_detection2 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        int2 coord = (int2)(get_global_id(0), get_global_id(1));
        const float4 in = read_imagef(src, clamp_sampler, coord);
        
        // uint width = get_global_size(0);
	    // uint height = get_global_size(1);
        


    //SMOOTHING
    float4 s = (float4)(0.0f);
    int i = 0;
    // //Half smoothing smoothing size / 2 (5/2)

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
            // s += filterWeights[i] * convert_float4(cur_pix);
            ++i;
        }
    }
    
        float harris_k = 0.05;
        float4 r = (float4)(0);
        r.x = ((s.x * s.y - s.z * s.z) - harris_k * (s.x + s.y) * (s.x + s.y));
        
        // r = fast_normalize(r);

        // if (r.x< 60) {
        //     r.x = 0;
        // }
        r.y = r.x;
        r.z = r.x;
        // write_imagef(dest, coord, Gx);
        write_imageui(dest, coord, convert_uint4(r));
        // }
    }

// __kernel void feature_detection3 (
//     __read_only image2d_t src,
//     __global float* row_max_values) {

//     const int row = get_global_id(0);
//     const int width = get_image_width(src);
//     float row_max = 0.0f;

//     for (int x = 0; x < width; ++x) {
//         const int2 pos = {x, row};
//         const float4 s = read_imagef(src, reflect_sampler, pos);
//         row_max = max(s.x, row_max);
//     }

//     row_max_values[row] = row_max;
// }