#ifndef STMAT_H
#define STMAT_H

namespace nml
{
	// The String Matrix for Numerical Method
	class stmat : public numem<std::string>
	{
		// Variables
	public:


		// Functions
	public:
		// Console out for debugging
		void cout() const;
		// Convert the matrix to the 'numem' data
		const numem<std::string> to_data() const;
		// Convert the data to the matrix
		static stmat to_mat(const numem<std::string>& data);
		static stmat to_mat(const std::vector<std::string>& vec);
		// Regenerate a matrix (nrows : the number of the new rows)
		static stmat to_mat(const std::vector<std::string>& vec, int nrows);
		// Clone the matrix
		const stmat clone() const;
		// Reshape the matrix (nrows : the number of the new rows)
		const stmat reshape(int nrows) const;
		// Vertically append the matrix
		static stmat vappend(const std::vector<stmat>& arr);
		const stmat vappend(const stmat& mat) const;
		// Horizontally append the matrix
		static stmat happend(const std::vector<stmat>& arr);
		const stmat happend(const stmat& mat) const;
		// Get the selected row matrix
		const stmat row(int idx) const;
		// Get the selected column matrix
		const stmat col(int idx) const;
		// Swap the matrix
		void swapr(int idx0, int idx1) const;
		void swapc(int idx0, int idx1) const;
		// Get a diagonal matrix
		const stmat diag() const;
		// Get a minor matrix except the selected row
		const stmat minorr(int idx) const;
		// Get a minor matrix except the selected column
		const stmat minorc(int idx) const;
		// Get a minor matrix except the selected row and column
		const stmat minor(int ridx, int cidx) const;
		// Get an unique matrix
		const stmat uniq() const;
		// Count the value in the matrix
		int count(std::string val) const;

		// Operators
	public:
		// Copy the matrix
		stmat& operator=(const stmat& mat);

		// Constructors & Destructor
	public:
		stmat();
		stmat(const msize& size);
		stmat(const msize& size, std::string val);
		stmat(const numem<std::string>& data);
		stmat(const stmat& obj);
		~stmat();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};
}

#endif