#ifndef MSIZE_H
#define MSIZE_H

namespace nml
{
	// Size for Memory
	class msize
	{
		// Variables
	public:
		int d0;		// 1st Dimension
		int d1;		// 2nd Dimension
		int d2;		// 3rd Dimension

		// Functions
	public:
		// Set Size Information
		void set(const int length);
		void set(const int rows, const int cols);
		void set(const int rows, const int cols, const int channels);

		// Operators
	public:
		msize& operator=(const msize& obj);

		// Constructors & Destructor
	public:
		msize();
		msize(const int length);
		msize(const int rows, const int cols);
		msize(const int rows, const int cols, const int channels);
		~msize();

	};
}

#endif