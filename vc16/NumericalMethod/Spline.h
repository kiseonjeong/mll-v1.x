#ifndef SPLINE_H
#define SPLINE_H

namespace nml
{
	// The Spline for Numerical Method
	class spline : public object
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		virtual void estimate(const numat& X, const numat& Y) = 0;
		// Interpolate the data
		virtual const double interpolate(const double x) const = 0;
		// Get the estimated coefficients
		virtual const numat coeff() const;

		// Variables
	protected:
		// Estimated coefficients
		numat C;
		// Interval matrix
		numat I;
		// Backup vector X
		numat X;
		// Backup vector Y
		numat Y;
		// Number of points
		int numpt;
		// Number of intervals
		int numit;

		// Functions
	protected:
		// Check the matrices are empty or not
		void checkEmpty(const numat& X, const numat& Y) const;
		// Check the length of the matrices
		void checkLength(const numat& X, const numat& Y) const;
		// Compare the length between the matrices
		void compareLength(const numat& X, const numat& Y) const;
		// Get an interval index
		const int getInterval(const double x) const;

	};

	// The Linear Spline
	class spline1d : public spline
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		virtual void estimate(const numat& X, const numat& Y);
		// Interpolate the data
		virtual const double interpolate(const double x) const;

		// Operators
	public:
		spline1d& operator=(const spline1d& obj);

		// Constructors & Destructor
	public:
		spline1d();
		spline1d(const numat& X, const numat& Y);
		spline1d(const spline1d& obj);
		virtual ~spline1d();

		// Variables
	protected:


		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const object& obj);
		// Clear the object
		virtual void clearObject();
		// Calculate the coefficients
		virtual void calculate();

		// Variables
	private:


		// Functions
	private:
		// Calculate the slope
		const double slope(const double x0, const double y0, const double x1, const double y1) const;

	};

	// The Quadratic Spline
	class spline2d : public spline1d
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		void estimate(const numat& X, const numat& Y);
		// Interpolate the data
		const double interpolate(const double x) const;

		// Operators
	public:
		spline2d& operator=(const spline2d& obj);

		// Constructors & Destructor
	public:
		spline2d();
		spline2d(const numat& X, const numat& Y);
		spline2d(const spline2d& obj);
		~spline2d();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Calculate the coefficients
		void calculate();

	};

	// The End Condition Type of the Cubic Spline
	typedef enum _cubic_type
	{
		CUBIC_UNKNOWN = 0,
		CUBIC_NATURAL,
		CUBIC_CLAMPED,
		CUBIC_NOTAKNOT,
	} cubic_type;

	// The Cubic Spline Parameters
	class cubic_params : public object
	{
		// Variables
	public:
		// End condition type
		prop::get<cubic_type> type;

		// Functions
	public:


		// Constructors & Destructor
	public:
		cubic_params();
		~cubic_params();

		// Variables
	protected:
		// End condition type
 		cubic_type _type;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const object& obj);
		// Clear the object
		virtual void clearObject();

	};

	// The Natural Type
	class natural : public cubic_params
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		// Copy Object
		natural& operator=(const natural& obj);

		// Constructors & Destructor
	public:
		natural();
		natural(const natural& obj);
		~natural();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Clamped Type
	class clamped : public cubic_params
	{
		// Variables
	public:


		// Functions
	public:


		// Operators
	public:
		// Copy Object
		clamped& operator=(const clamped& obj);

		// Constructors & Destructor
	public:
		clamped();
		clamped(const clamped& obj);
		~clamped();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Not a Knot Type
	class notaknot : public cubic_params
	{
		// Variables
	public:
		// Condition parameters
		prop::get<double> alpha;
		prop::get<double> beta;

		// Functions
	public:
		// Set the parameters
		void set(const double alpha, const double beta);

		// Operators
	public:
		// Copy the object
		notaknot& operator=(const notaknot& obj);

		// Constructors & Destructor
	public:
		notaknot();
		notaknot(const double alpha, const double beta);
		notaknot(const notaknot& obj);
		~notaknot();

		// Variables
	private:
		// Condition parameters
		double _alpha;
		double _beta;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();

	};

	// The Cubic Spline
	class spline3d : public spline1d
	{
		// Variables
	public:


		// Functions
	public:
		// Set an end condition for the cubic spline
		void condition(const cubic_params& params);
		// Estimate the coefficients
		void estimate(const numat& X, const numat& Y);
		// Interpolate the data
		const double interpolate(const double x) const;

		// Operators
	public:
		spline3d& operator=(const spline3d& obj);

		// Constructors & Destructor
	public:
		spline3d();
		spline3d(const cubic_params& params);
		spline3d(const numat& X, const numat& Y, const cubic_params& params = natural());
		spline3d(const spline3d& obj);
		~spline3d();

		// Variables
	private:
		// Cubic spline parameters
		cubic_params* params;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Create the parameters
		void createParams(const cubic_params& params);
		// Calculate the coefficients
		void calculate();
		// Get a condition matrix
		const numat condmat(const numat& W) const;
		// Get a result matrix
		const numat resmat(const numat& S) const;

	};

	// Rename The Spliners
	typedef spline1d linear_spline;
	typedef spline2d quadratic_spline;
	typedef spline3d cubic_spline;

	// The Hermite Spline Type
	typedef enum _hermite_type
	{
		HERMITE_UNKNOWN = 0,
		HERMITE_FINITE_DIFFERENCE,
		HERMITE_CARDINAL,
		HERMITE_CATMULL_ROM,
	} hermite_type;

	// The Hermite Spline Parameters
	class hermite_params : public object
	{
		// Variables
	public:
		// Hermite spline type
		prop::get<hermite_type> type;

		// Functions
	public:
		// Calculate the coefficients
		virtual const numat calculate(const numat& X, const numat& Y, const int numpt) const = 0;

		// Constructors & Destructor
	public:
		hermite_params();
		~hermite_params();

		// Variables
	protected:
		// Hermite spline type
		hermite_type _type;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const object& obj);
		// Clear the object
		virtual void clearObject();

	};

	// The Finite Difference Type
	class finitediff : public hermite_params
	{
		// Variables
	public:


		// Functions
	public:
		// Calculate the coefficents
		virtual const numat calculate(const numat& X, const numat& Y, const int numpt) const;

		// Operators
	public:
		// Copy the object
		finitediff& operator=(const finitediff& obj);

		// Constructors & Destructor
	public:
		finitediff();
		finitediff(const finitediff& obj);
		virtual ~finitediff();

		// Variables
	protected:


		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Calculate the slope
		const double slope(const double x0, const double y0, const double x1, const double y1) const;

	};

	// The Cardinal Type
	class cardinal : public finitediff
	{
		// Variables
	public:
		// Tension Parameter
		prop::get<double> c;

		// Functions
	public:
		// Set the parameters
		void set(double c);
		// Calculate the coefficents
		virtual const numat calculate(const numat& X, const numat& Y, const int numpt) const;

		// Operators
	public:
		// Copy the object
		cardinal& operator=(const cardinal& obj);

		// Constructors & Destructor
	public:
		cardinal();
		cardinal(double c);
		cardinal(const cardinal& obj);
		virtual ~cardinal();

		// Variables
	protected:
		// Tension parameter
		double _c;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		void copyObject(const object& obj);

	};

	// The Catmull-Rom Type
	class catmullrom : public cardinal
	{
		// Variables
	public:


		// Functions
	public:
		// Calculate the coefficents
		const numat calculate(const numat& X, const numat& Y, const int numpt) const;

		// Operators
	public:
		// Copy the object
		catmullrom& operator=(const catmullrom& obj);

		// Constructors & Destructor
	public:
		catmullrom();
		catmullrom(const catmullrom& obj);
		~catmullrom();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};

	// The Hermite Spline
	class hermite : public spline
	{
		// Variables
	public:


		// Functions
	public:
		// Set a spline condition for the hermite spline
		void condition(const hermite_params& params);
		// Estimate the coefficients
		void estimate(const numat& X, const numat& Y);
		// Interpolate the data
		const double interpolate(const double x) const;

		// Operators
	public:
		hermite& operator=(const hermite& obj);

		// Constructors & Destructor
	public:
		hermite();
		hermite(const hermite_params& params);
		hermite(const numat& X, const numat& Y, const hermite_params& params = finitediff());
		hermite(const hermite& obj);
		~hermite();

		// Variables
	private:
		// Hermite spline parameters
		hermite_params* params;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Create the hermite parameters
		void createParams(const hermite_params& params);
		// Calculate a basis
		const double basis(const double x, const int idx, const int mode) const;
		// Calculate a tangent
		const double tangent(const double x, const int idx) const;

	};

	// The Akima Spline
	class akima : public spline
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		void estimate(const numat& X, const numat& Y);
		// Interpolate the data
		const double interpolate(const double x) const;

		// Operators
	public:
		akima& operator=(const akima& obj);

		// Constructors & Destructor
	public:
		akima();
		akima(const numat& X, const numat& Y);
		akima(const akima& obj);
		~akima();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Calculate the coefficients
		void calculate();
		// Augment the slopes on the end points
		const numat augment(const numat& S) const;
		// Calculate a derivative value
		const double derivative(const numat& S, const int idx) const;

	};
}

#endif