#ifndef REGRESSION_H
#define REGRESSION_H

namespace nml
{
	// The Regression for Numerical Method
	class regression : public object
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		virtual void estimate(const numat& X, const numat& Y, const int deg) = 0;
		// Interpolate the data
		virtual const double fit(const double x) const = 0;
		// Get the estimated coefficients
		virtual const numat coeff() const;

		// Variables
	protected:
		// Estimated coefficients
		numat C;
		// Degree of regression
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

	// The Curve Fitting using Ordinary Least Square
	class curvefit : public regression
	{
		// Variables
	public:


		// Functions
	public:
		// Estimate the coefficients
		void estimate(const numat& X, const numat& Y, const int deg);
		// Interpolate the data
		const double fit(const double x) const;

		// Operators
	public:
		// Copy the object
		curvefit& operator=(const curvefit& obj);

		// Constructors & Destructor
	public:
		curvefit();
		curvefit(const numat& X, const numat& Y, const int deg);
		curvefit(const curvefit& obj);
		~curvefit();

		// Variables
	private:
		// Epsilon parameter
		double epsilon;
		// Whitening information on the vector X
		double meanX;
		double stdX;
		// Whitening information on the vector Y
		double meanY;
		double stdY;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Do whitening on the vectors
		void whitening();
		// Calculate the coefficients
		void calculate();

	};
}

#endif