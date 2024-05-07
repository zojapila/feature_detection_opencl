__constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_NEAREST;
__constant sampler_t clamp_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;


__kernel void feature_detection (
    __read_only image2d_t src,
    __write_only image2d_t dest) {


    // ARG32TOFLOAT  
    const int2 pos = {get_global_id(0), get_global_id(1)};
    const float4 in = read_imagef(src, clamp_sampler, pos);
    float4 pix = (float4)(in.x * 0.2989f + in.y * 0.5870f + in.z * 0.1140f);

    //SMOOTHING
    float4 sum = (float4)(0.0f);
    int i = 0;
    //Half smoothing smoothing size / 2 (5/2)

    //filterWeights is gaussianfilter
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
            float4 cur_pix = read_imagef(src, reflect_sampler, pos + (int2)(x,y));
            float4 cur_pix_gray = (float4)(cur_pix.x * 0.2989f + cur_pix.y * 0.5870f + cur_pix.z * 0.1140f);

            sum += filterWeights[i] * cur_pix_gray;
            ++i;
        }
    }
    
    write_imagef(dest, pos, sum);

}

__kernel void feature_detection2 (
    __read_only image2d_t src,
    __write_only image2d_t dest) {
        const int2 pos = {get_global_id(0), get_global_id(1)};
        // const float4 in = read_imagef(src, clamp_sampler, pos);
        float4 s = (float4)(0.0f);
        int half_structure = 2;
        for (int y = -half_structure; y <= half_structure; ++y) 
        {
            for (int x = -half_structure; x <= half_structure; ++x) 
            {
            const int2 window_pos = pos + (int2)(x,y);
            float4 sumx = read_imagef(src, reflect_sampler, window_pos - (int2)(1,0)) - read_imagef(src, reflect_sampler, window_pos + (int2)(1,0));
            float4 sumy = read_imagef(src, reflect_sampler, window_pos - (int2)(0,1)) - read_imagef(src, reflect_sampler, window_pos + (int2)(0,1));
            s.x += sumx.x *sumx.x;
            s.y += sumy.x *sumy.x;
            s.z += sumx.x *sumy.x;
            }
        }
        float harris_k = 0.05;
        const float4 r;
        r.x = (s.x * s.y - s.z * s.z) - harris_k * (s.x + s.y) * (s.x + s.y);

        write_imagef(dest, pos, r);
    }
