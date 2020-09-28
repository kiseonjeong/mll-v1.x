#ifndef DECOMPOSITION_H
#define DECOMPOSITION_H

namespace nml
{
	// The Decomposition for Algebra
	class decomposition : public object
	{
		// Variables
	public:


		// Functions
	public:
		// Decompose the matrix
		virtual const std::vector<numat> decomp(const numat& mat) = 0;
		// Get the decomposition result
		const std::vector<numat> result() const;

		// Operators
	public:
		// Get the decomposition result
		virtual const numat operator[](const int idx) const;

		// Variables
	protected:
		// Decomposition result
		std::vector<numat> D;

		// Functions
	protected:
		// Check the matrix is empty or not
		void checkEmpty(const numat& mat) const;
		// Check the matrix is square or not
		void checkSquareness(const numat& mat) const;

	};

	// The pivot mode for the LUD
	typedef enum _pivotMode
	{
		LU_PARTIAL_PIVOT = 0,
		LU_FULL_PIVOT,
	} pivotMode;

	// The LU decomposition
	class LUD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		// Set a pivot mode
		void pivot(const pivotMode mode);
		// Decompose the matrix
		const std::vector<numat> decomp(const numat& mat);

		// Operators
	public:
		// Copy the object
		LUD& operator=(const LUD& obj);
		
		// Constructors & Destructor
	public:
		LUD();
		LUD(const numat& mat);
		LUD(const numat& mat, const pivotMode mode);
		LUD(const LUD& obj);
		~LUD();

		// Variables
	private:
		// Pivot mode flag
		pivotMode mode;
		// Factorization matrices
		std::vector<numat> P;
		std::vector<numat> M;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Factorize the matrix
		void factorize(const numat& mat);

	};

	// The QR decomposition
	class QRD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		// Decompose the matrix
		const std::vector<numat> decomp(const numat& mat);

		// Operators
	public:
		// Copy the object
		QRD& operator=(const QRD& obj);

		// Constructors & Destructor
	public:
		QRD();
		QRD(const numat& mat);
		QRD(const QRD& obj);
		~QRD();

		// Variables
	private:
		// Transformation matrix
		std::vector<numat> T;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Reduce the matrix
		const numat reduction(const numat& mat, const int flag) const;
		// Convert the matrix using House Holder Transform
		const numat houseHolder(const numat& mat) const;
		const numat reflection(const numat& mat, const int length) const;
		// Calculate a vector length
		const double vecsize(const numat& mat) const;
		// Extend a matrix
		const numat extension(const numat& mat, const int offset, const int length) const;

	};

	// The EV decomposition
	class EVD : public decomposition
	{
		// Variables
	public:


		// Functions
	public:
		// Decompose the matrix
		virtual const std::vector<numat> decomp(const numat& mat);

		// Operators
	public:
		// Copy the object
		EVD& operator=(const EVD& obj);

		// Constructors & Destructor
	public:
		EVD();
		EVD(const numat& mat);
		EVD(const EVD& obj);
		virtual ~EVD();

		// Variables
	protected:

		
		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		void copyObject(const object& obj);
		// Clear the object
		void clearObject();
		// Check the matrix is convergence or not
		const bool isConvergence(const numat& curr, const numat& prev) const;

		// Variables
	private:


		// Functions
	private:

	};

	// The SV decomposition
	class SVD : public EVD
	{
		// Variables
	public:


		// Functions
	public:
		// Decompose the matrix
		const std::vector<numat> decomp(const numat& mat);
		// Get a pseudo inverse matrix
		static numat pinv(const numat& mat);

		// Operators
	public:
		// Copy the object
		SVD& operator=(const SVD& obj);
		// Get the decomposition result
		const numat operator[](const int idx) const;

		// Constructors & Destructor
	public:
		SVD();
		SVD(const numat& mat);
		SVD(const SVD& obj);
		~SVD();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();

	};
}

#endif