#ifndef NUMEM_H
#define NUMEM_H

namespace nml
{
	// The Memory for Numerical Method
	template <typename var> class numem : public object
	{
		// Variables
	public:
		// Data row information
		prop::get<int> rows;
		// Data column information
		prop::get<int> cols;

		// Functions
	public:
		// Get the data pointer
		var* ptr() const;
		// Set the data pointer
		void ptr(const msize& size, var* data);
		// Check the data is empty or not
		bool empty() const;
		// Release the data
		void release();
		// Create the data
		void create(const msize& size);
		void create(const msize& size, var val);
		// Copy the data
		void copy(const numem& obj);
		void copy(const msize& size, const var* &data);
		// Get the data length
		int length() const;

		// Operators
	public:
		numem& operator=(const numem& obj);
		var& operator[](const int idx) const;
		var& operator()(const int idx) const;
		var& operator()(int ridx, int cidx) const;

		// Constructors & Destructor
	public:
		numem();
		numem(const msize& size);
		numem(const msize& size, var val);
		numem(const numem& obj);
		virtual ~numem();		

		// Variables
	protected:
		// Data for Numerical Method
		int _rows;
		int _cols;
		var* _ptr;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const object& obj);
		// Clear the object
		virtual void clearObject();

		// Variables
	private:


		// Functions
	private:
		// Overwrite the data
		void overwrite(int rows, int cols);

	};
}

#include "Numem.hpp"

#endif