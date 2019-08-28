#ifndef INTRINSICS_H__
#define INTRINSICS_H__

struct Intrinsics {
	Intrinsics() : fx(0), fy(0), cx(0), cy(0) {
	}

	Intrinsics(float fx_, float fy_, float cx_, float cy_) :
		fx(fx_), fy(fy_), cx(cx_), cy(cy_) {
	}

	Intrinsics operator()(int scale) {
		int i = 1 << scale;
		return Intrinsics(fx / i, fy / i, cx / i, cy / i);
	}

	float fx, fy, cx, cy;
};

#endif
