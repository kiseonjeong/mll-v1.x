#ifndef INTERPOLATION_H
#define INTERPOLATION_H

namespace nml
{
	// The Interpolation for Numerical Method
	class interpolation : public object
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
		// Degree of interpolation
		int deg;
		// Backup vector X
		numat X;
		// Backup vector Y
		numat Y;

		// Functions
	protected:
		// Check the matrices are empty or not
		void checkEmpty(const numat& X, const numat& Y) const;
		// Check the length of the matrices
		void checkLength(const numat& X, const numat& Y) const;
		// Compare the length between the matrices
		void compareLength(const numat& X, const numat& Y) const;

	};

	// The Vandermonde Interpolation
	class vandermonde : public interpolation
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
		vandermonde& operator=(const vandermonde& obj);

		// Constructors & Destructor
	public:
		vandermonde();
		vandermonde(const numat& X, const numat& Y);
		vandermonde(const vandermonde& obj);
		~vandermonde();

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

	};

	// The Newton Interpolation
	class newton : public interpolation
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
		newton& operator=(const newton& obj);

		// Constructors & Destructor
	public:
		newton();
		newton(const numat& X, const numat& Y);
		newton(const newton& obj);
		~newton();

		// Variables
	private:
		// Index of coefficients
		int idx;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Calculate the coefficients
		void calculate(const numat& X, const numat& Y);
		// Calculate the slope
		const double slope(const double x0, const double y0, const double x1, const double y1) const;

	};

	// Lagrange Interpolation
	class lagrange : public interpolation
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate Coefficients
		void estimate(const numat& X, const numat& Y);
		// Interpolate Data
		const double interpolate(const double x) const;

		// Operators
	public:
		lagrange& operator=(const lagrange& obj);

		// Constructors & Destructor
	public:
		lagrange();
		lagrange(const numat& X, const numat& Y);
		lagrange(const lagrange& obj);
		~lagrange();

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
		const double calculate(const double x, const int idx) const;
		// Get the estimated coefficients (Prohibit this access)
		const numat coeff() const;

	};
}

#endif