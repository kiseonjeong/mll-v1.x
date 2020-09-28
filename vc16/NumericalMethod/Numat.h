#ifndef NUMAT_H
#define NUMAT_H

namespace nml
{
	// The Numerical Matrix for Numerical Method
	class numat : public numem<double>
	{
		// Variables
	public:


		// Functions
	public:
		// Console out for debugging
		void cout() const;
		// Convert the matrix to the 'numem' data
		const numem<double> to_data() const;
		// Convert the data to the matrix
		static numat to_mat(const numem<double>& data);
		static numat to_mat(const std::vector<double>& vec);
		// Regenerate a matrix (nrows : the number of the new rows)
		static numat to_mat(const std::vector<double>& vec, int nrows);
		// Clone the matrix
		const numat clone() const;
		// Reshape the matrix (nrows : the number of the new rows)
		const numat reshape(int nrows) const;
		// Vertically append the matrix
		static numat vappend(const std::vector<numat>& arr);
		const numat vappend(const numat& mat) const;
		// Horizontally append the matrix
		static numat happend(const std::vector<numat>& arr);
		const numat happend(const numat& mat) const;
		// Get the selected row matrix
		const numat row(int idx) const;
		const numat row(int bi, int ei, int ii = -1) const;
		// Get the selected column matrix
		const numat col(int idx) const;
		const numat col(int bi, int ei, int ii = -1) const;
		// Swap the matrix
		void swapr(int idx0, int idx1) const;
		void swapc(int idx0, int idx1) const;
		// Get a diagonal matrix
		const numat diag() const;
		// Get a values matrix
		static numat values(msize& size, double val);
		// Get a zeros matrix
		static numat zeros(msize& size);
		// Get an ones matrix
		static numat ones(msize& size);
		// Get an eyes matrix
		static numat eyes(int length);
		// Get a permutation matrix
		static numat permut(int length, int idx0, int idx1);
		// Get a minor matrix except the selected row
		const numat minorr(int idx) const;
		// Get a minor matrix except the selected column
		const numat minorc(int idx) const;
		// Get a minor matrix except the selected row and column
		const numat minor(int ridx, int cidx) const;
		// Get an unique matrix
		const numat uniq() const;
		// Count the value in the matrix
		int count(double val) const;
		// Compare the value
		static bool almost(double val, double tar, int round = 30);
		// Sort the matrix
		void sortr(bool descending = false) const;
		void sortc(bool descending = false) const;
		void sort(bool descending = false) const;
		// Find a minimum matrix
		static numat minr(const numat& mat);
		static numat minc(const numat& mat);
		static double min(const numat& mat);
		// Find a maximum matrix
		static numat maxr(const numat& mat);
		static numat maxc(const numat& mat);
		static double max(const numat& mat);
		// Calculate a sum matrix
		static numat sumr(const numat& mat);
		static numat sumc(const numat& mat);
		static double sum(const numat& mat);
		// Calculate a mean matrix
		static numat meanr(const numat& mat);
		static numat meanc(const numat& mat);
		static double mean(const numat& mat);
		// Calculate a variance matrix
		static numat varr(const numat& mat);
		static numat varc(const numat& mat);
		static double var(const numat& mat);
		// Calculate a standard deviation matrix
		static numat stdr(const numat& mat);
		static numat stdc(const numat& mat);
		static double std(const numat& mat);
		// Calculate a covariance matrix
		const numat cov() const;
		// Calculate the mathematical operations
		void log() const;
		void log(double base) const;
		static numat log(const numat& mat);
		static numat log(const numat& mat, double base);
		void exp() const;
		static numat exp(const numat& mat);
		void pow(double p) const;
		static numat pow(const numat& mat, double p);
		void sqrt() const;
		static numat sqrt(const numat& mat);
		void abs() const;
		static numat abs(const numat& mat);
		// Calculate an addition matrix
		const numat add(const numat& mat) const;
		const numat add(double val) const;
		// Calculate a subtraction matrix
		const numat sub(const numat& mat) const;
		const numat sub(double val) const;
		// Calculate a multiplication matrix
		const numat mul(const numat& mat) const;
		const numat mul(double val) const;
		// Calculate a division matrix
		const numat div(const numat& mat) const;
		const numat div(double val) const;
		// Get a transpose matrix
		const numat trans() const;
		const numat T() const;
		// Get a dot product matrix
		const numat dot(const numat& mat) const;
		// Get a cofactor matrix
		const numat cof() const;
		// Get an inverse matrix
		const numat inv() const;
		const numat I() const;
		// Calculate a determinant
		const double det() const;

		// Operators
	public:
		// Copy the matrix
		numat& operator=(const numat& mat);

		// Calculate an addition matrix
		const numat operator+(const numat& mat) const;
		const numat operator+(double val) const;
		numat& operator+=(const numat& mat);
		numat& operator+=(double val);
		friend const numat operator+(double val, const numat& mat)
		{
			// Calculate addition operations
			numat out = mat;
			for (int i = 0; i < out.length(); i++)
			{
				out(i) = val + out(i);
			}

			return out;
		}

		// Calculate a subtraction matrix
		const numat operator-(const numat& mat) const;
		const numat operator-(double val) const;
		numat& operator-=(const numat& mat);
		numat& operator-=(double val);
		friend const numat operator-(double val, const numat& mat)
		{
			// Calculate subtraction operations
			numat out = mat;
			for (int i = 0; i < out.length(); i++)
			{
				out(i) = val - out(i);
			}

			return out;
		}

		// Calculate a multiplication matrix
		const numat operator*(const numat& mat) const;
		const numat operator*(double val) const;
		numat& operator*=(const numat& mat);
		numat& operator*=(double val);
		friend const numat operator*(double val, const numat& mat)
		{
			// Calculate multiplication operations
			numat out = mat;
			for (int i = 0; i < out.length(); i++)
			{
				out(i) = val * out(i);
			}

			return out;
		}

		// Calculate a division matrix
		const numat operator/(const numat& mat) const;
		const numat operator/(double val) const;
		numat& operator/=(const numat& mat);
		numat& operator/=(double val);
		friend const numat operator/(double val, const numat& mat)
		{
			// Check the value
			assert(val != 0.0);

			// Calculate division operations
			numat out = mat;
			for (int i = 0; i < out.length(); i++)
			{
				out(i) = val / out(i);
			}

			return out;
		}

		// Convert the sign on the matrix
		friend const numat operator-(const numat& mat)
		{
			numat out = mat;
			for (int i = 0; i < out.length(); i++)
			{
				out(i) = -out(i);
			}

			return out;
		}

		// Constructors & Destructor
	public:
		numat();
		numat(const msize& size);
		numat(const msize& size, double val);
		numat(const numem<double>& data);
		numat(const numat& obj);
		~numat();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Expand a matrix for a cofactor matrix
		const double expansion(const numat& mat) const;

	};
}

#endif