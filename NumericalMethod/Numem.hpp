namespace nml
{
	template<typename var> numem<var>::numem() : rows(_rows), cols(_cols)
	{
		// Set an object
		setObject();
	}

	template<typename var> numem<var>::numem(const msize& size) : rows(_rows), cols(_cols)
	{
		// Set an object
		setObject();

		// Create the data
		create(size);
	}

	template<typename var> numem<var>::numem(const msize& size, var val) : rows(_rows), cols(_cols)
	{
		// Set an object
		setObject();

		// Create the data
		create(size, val);
	}

	template<typename var> numem<var>::numem(const numem& obj) : rows(_rows), cols(_cols)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	template<typename var> numem<var>::~numem()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> numem<var>& numem<var>::operator=(const numem& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	template<typename var> var& numem<var>::operator[](const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < _rows * _cols);

		return _ptr[idx];
	}

	template<typename var> var& numem<var>::operator()(const int idx) const
	{
		// Check the index
		assert(idx >= 0 && idx < _rows * _cols);

		return _ptr[idx];
	}

	template<typename var> var& numem<var>::operator()(int ridx, int cidx) const
	{
		// Check the indices
		assert(ridx >= 0 && ridx < _rows);
		assert(cidx >= 0 && cidx < _cols);

		return _ptr[ridx * _cols + cidx];
	}

	template<typename var> void numem<var>::setObject()
	{
		// Set the parameters
		setType(*this);
		_rows = -1;
		_cols = -1;

		// Set the memories
		_ptr = nullptr;
	}

	template<typename var> void numem<var>::copyObject(const object& obj)
	{
		// Do down casting
		const numem<var>* _obj = static_cast<const numem<var>*>(&obj);

		// Check the object status
		if (_obj->empty() == false)
		{
			// Copy the parameters
			overwrite(_obj->_rows, _obj->_cols);

			// Copy the memories
			for (int i = 0; i < _obj->_rows * _obj->_cols; i++)
			{
				_ptr[i] = _obj->_ptr[i];
			}
		}
	}

	template<typename var> void numem<var>::clearObject()
	{
		// Clear the parameters
		_rows = -1;
		_cols = -1;

		// Clear the memories
		if (empty() == false)
		{
			delete[] _ptr;
			_ptr = nullptr;
		}
	}

	template<typename var> void numem<var>::overwrite(int rows, int cols)
	{
		// Check the size information
		assert(rows >= 0 && cols >= 0);

		// Release the old data
		release();

		// Create an object
		_rows = rows;
		_cols = cols;
		_ptr = new var[rows * cols];
	}

	template<typename var> var* numem<var>::ptr() const
	{
		// Get the data pointer
		return _ptr;
	}

	template<typename var> void numem<var>::ptr(const msize& size, var* data)
	{
		// Check the size information
		assert(rows >= 0 && cols >= 0);

		// Release the old data
		release();

		// Create an object
		_rows = size.d0;
		_cols = size.d1;

		// Set the data
		_ptr = data;
	}

	template<typename var> bool numem<var>::empty() const
	{
		// Check the data pointer
		if (_ptr == nullptr)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	template<typename var> void numem<var>::release()
	{
		// Clear the object
		clearObject();
	}

	template<typename var> void numem<var>::create(const msize& size)
	{
		// Overwrite the data
		overwrite(size.d0, size.d1);
	}

	template<typename var> void numem<var>::create(const msize& size, var val)
	{
		// Overwrite the data
		overwrite(size.d0, size.d1);

		// Initialize the data
		for (int i = 0; i < rows * cols; i++)
		{
			_ptr[i] = val;
		}
	}

	template<typename var> void numem<var>::copy(const numem& obj)
	{
		// Copy the data
		copyObject(obj);
	}

	template<typename var> void numem<var>::copy(const msize& size, const var* &data)
	{
		// Overwrite the data
		overwrite(size.d0, size.d1);

		// Initialize the data
		for (int i = 0; i < rows * cols; i++)
		{
			_ptr[i] = data[i];
		}
	}

	template<typename var> int numem<var>::length() const
	{
		// Check the data and get the length information
		if (empty() == true)
		{
			return -1;
		}
		else
		{
			return _rows * _cols;
		}
	}
}