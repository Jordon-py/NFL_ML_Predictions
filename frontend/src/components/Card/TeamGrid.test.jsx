import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';

// We'll mock the api client for the fallback predictGame call in one test.
// The mock is configured per-test below when needed.

import '@testing-library/jest-dom';

const sampleGame = {
      season: 2025,
      week: 12,
      home_team: 'HOU',
      away_team: 'BUF',
      home_abbr: 'HOU',
      away_abbr: 'BUF',
};

describe( 'TeamGrid', () =>
{
      it( 'calls onPredict when provided and a card is clicked', async () =>
      {
            // Ensure module cache is clean and import the component fresh
            vi.resetModules();
            const { default: TeamGrid } = await import( './TeamGrid.jsx' );
            const onPredict = vi.fn();
            const user = userEvent.setup();

            render(
                  <TeamGrid
                        games={ [ sampleGame ] }
                        week={ 12 }
                        teams={ {} }
                        predictions={ {} }
                        loading={ {} }
                        errors={ {} }
                        onPredict={ onPredict }
                  />
            );

            const btns = await screen.findAllByRole( 'button' );
            await user.click( btns[ 0 ] );

            expect( onPredict ).toHaveBeenCalledTimes( 1 );
            expect( onPredict ).toHaveBeenCalledWith( sampleGame );
      } );

      it( 'falls back to predictGame and shows loading while request is inflight', async () =>
      {
            // Create a deferred promise so we can assert the loading state while it is pending
            let resolveDeferred;
            const deferred = new Promise( ( res ) => { resolveDeferred = res; } );

            // Mock the module so TeamGrid's internal predictGame will return our deferred promise
            vi.mock( '../../api/client', () => ( {
                  predictGame: vi.fn( () => deferred ),
            } ) );

            // Re-import the component (ensure mocks take effect). In vitest this import will use the mocked module.
            // Reset modules so the vi.mock above is applied when importing
            vi.resetModules();
            const TeamGridFallback = ( await import( './TeamGrid.jsx' ) ).default;

            const user = userEvent.setup();

            render(
                  <TeamGridFallback
                        games={ [ sampleGame ] }
                        week={ 12 }
                        teams={ {} }
                        predictions={ {} }
                        loading={ {} }
                        errors={ {} }
                  />
            );

            const btns = await screen.findAllByRole( 'button' );
            // Click the card to trigger the fallback predictGame
            await user.click( btns[ 0 ] );

            // While the deferred promise is unresolved, the card should show a loading state.
            expect( screen.getByText( /Fetching prediction/i ) ).toBeInTheDocument();

            // Resolve the deferred promise and then wait for the loading state to disappear
            resolveDeferred( {} );
            await waitFor( () => expect( screen.queryByText( /Fetching prediction/i ) ).not.toBeInTheDocument() );
      } );
} );
